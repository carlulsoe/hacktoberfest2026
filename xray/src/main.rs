//! LLM X-ray: watch a model pick tokens one at a time, and meddle with it.
//!
//!   xray "Write a haiku about the borrow checker."
//!   xray --schema person.json "Describe a person."   # j = schema mode

use std::time::{Duration, Instant};

use anyhow::Result;
use engine::{
    Engine, Rng, Sampling,
    json::{JsonState, Schema},
};
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
};

const SYSTEM: &str = "You are a helpful assistant.";
const DEFAULT_PROMPT: &str = "Write a haiku about Rust's borrow checker.";
const SHOWN: usize = 12;

struct Step {
    id: u32,
    /// What the model thought of this token (before masking/forcing).
    prob: f32,
    forced: bool,
}

/// The model's view of the next token.
struct View {
    probs: Vec<f32>,
    mask: Option<Vec<bool>>,
    /// Candidate token ids to list, most probable first.
    shown: Vec<usize>,
    /// Share of probability the JSON mask allows.
    allowed_mass: f32,
    entropy_bits: f32,
    millis: u128,
}

struct App {
    engine: Engine,
    prompt: String,
    prompt_ids: Vec<u32>,
    steps: Vec<Step>,
    sampling: Sampling,
    json_mode: bool,
    /// What JSON mode enforces: any object unless `--schema` was given.
    schema: Schema,
    has_schema: bool,
    json: JsonState,
    view: View,
    /// Raw logits behind `view`, used for sampling.
    logits_cache: Vec<f32>,
    cursor: usize,
    autoplay: bool,
    done: bool,
    editing: Option<String>,
    message: String,
    rng: Rng,
}

impl App {
    fn new(engine: Engine, prompt: String, schema: Option<Schema>) -> Result<Self> {
        let empty = View { probs: vec![], mask: None, shown: vec![], allowed_mass: 1.0, entropy_bits: 0.0, millis: 0 };
        let mut app = Self {
            engine,
            prompt: String::new(),
            prompt_ids: vec![],
            steps: vec![],
            sampling: Sampling::default(),
            json_mode: false,
            has_schema: schema.is_some(),
            schema: schema.unwrap_or_default(),
            json: JsonState::default(),
            view: empty,
            logits_cache: vec![],
            cursor: 0,
            autoplay: false,
            done: false,
            editing: None,
            message: String::new(),
            rng: Rng::new(0x5eed),
        };
        app.restart(prompt)?;
        Ok(app)
    }

    fn restart(&mut self, prompt: String) -> Result<()> {
        self.prompt_ids = self.engine.chat_prompt(SYSTEM, &prompt)?;
        self.prompt = prompt;
        self.steps.clear();
        self.json = JsonState::default();
        self.done = false;
        self.refresh()
    }

    fn sequence(&self) -> Vec<u32> {
        self.prompt_ids.iter().copied().chain(self.steps.iter().map(|s| s.id)).collect()
    }

    /// Run the model on the current sequence and rebuild the view.
    fn refresh(&mut self) -> Result<()> {
        let start = Instant::now();
        let logits = self.engine.logits(&self.sequence())?;
        let probs = engine::probs(&logits, self.sampling.temperature, None);
        let mask = self.json_mode.then(|| self.engine.json_mask(&self.schema, &self.json));
        let mut shown = engine::top_n(&probs, SHOWN);
        let mut allowed_mass = 1.0;
        if let Some(m) = &mask {
            allowed_mass = probs.iter().zip(m).filter(|(_, ok)| **ok).map(|(p, _)| p).sum();
            // Make sure the best *allowed* tokens are visible too.
            let allowed = engine::probs(&logits, self.sampling.temperature, Some(m));
            for i in engine::top_n(&allowed, 4) {
                if m[i] && !shown.contains(&i) {
                    shown.push(i);
                }
            }
            shown.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        }
        let entropy_bits = -probs.iter().filter(|&&p| p > 0.0).map(|p| p * p.log2()).sum::<f32>();
        self.view = View { probs, mask, shown, allowed_mass, entropy_bits, millis: start.elapsed().as_millis() };
        self.cursor = self.cursor.min(self.view.shown.len().saturating_sub(1));
        self.logits_cache = logits;
        Ok(())
    }

    fn allowed(&self, id: usize) -> bool {
        self.view.mask.as_ref().is_none_or(|m| m[id])
    }

    /// Append a token: sampled if `forced` is None.
    fn step(&mut self, forced: Option<u32>) -> Result<()> {
        if self.done {
            return Ok(());
        }
        let id = match forced {
            Some(id) => id,
            None => engine::sample(&self.logits_cache, self.sampling, self.view.mask.as_deref(), &mut self.rng),
        };
        if self.engine.is_eos(id) {
            self.done = true;
            self.autoplay = false;
            return Ok(());
        }
        if self.json_mode {
            for &b in self.engine.token_bytes(id).unwrap_or_default() {
                self.json.feed(&self.schema, b);
            }
        }
        self.steps.push(Step { id, prob: self.view.probs[id as usize], forced: forced.is_some() });
        self.refresh()
    }

    fn undo(&mut self) -> Result<()> {
        if self.steps.pop().is_some() || self.done {
            self.done = false;
            self.json = self.replay_json().unwrap_or_default();
            self.refresh()?;
        }
        Ok(())
    }

    /// JSON parser state for the text so far, if it is a valid JSON prefix.
    fn replay_json(&self) -> Option<JsonState> {
        let mut s = JsonState::default();
        let ids: Vec<u32> = self.steps.iter().map(|s| s.id).collect();
        self.engine.decode(&ids).bytes().all(|b| s.feed(&self.schema, b)).then_some(s)
    }

    fn toggle_json(&mut self) -> Result<()> {
        if self.json_mode {
            self.json_mode = false;
        } else if let Some(s) = self.replay_json() {
            self.json = s;
            self.json_mode = true;
        } else {
            self.message = "Text so far doesn't fit the JSON constraint; undo or restart first.".into();
            return Ok(());
        }
        self.refresh()
    }

    /// Returns false to quit.
    fn handle(&mut self, code: KeyCode) -> Result<bool> {
        self.message.clear();
        if let Some(buf) = &mut self.editing {
            match code {
                KeyCode::Enter => {
                    let prompt = std::mem::take(buf);
                    self.editing = None;
                    self.restart(prompt)?;
                }
                KeyCode::Esc => self.editing = None,
                KeyCode::Backspace => _ = buf.pop(),
                KeyCode::Char(c) => buf.push(c),
                _ => {}
            }
            return Ok(true);
        }
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(false),
            KeyCode::Char(' ') | KeyCode::Right => self.step(None)?,
            KeyCode::Enter => {
                let id = self.view.shown[self.cursor];
                if self.allowed(id) {
                    self.step(Some(id as u32))?;
                } else {
                    self.message = "That token is masked by JSON mode.".into();
                }
            }
            KeyCode::Up => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Down => self.cursor = (self.cursor + 1).min(self.view.shown.len() - 1),
            KeyCode::Backspace | KeyCode::Left => self.undo()?,
            KeyCode::Char('a') => self.autoplay = !self.autoplay && !self.done,
            KeyCode::Char('j') => self.toggle_json()?,
            KeyCode::Char('r') => self.restart(self.prompt.clone())?,
            KeyCode::Char('e') => {
                self.autoplay = false;
                self.editing = Some(String::new());
            }
            KeyCode::Char('t') => self.sampling.temperature = (self.sampling.temperature - 0.1).max(0.0),
            KeyCode::Char('T') => self.sampling.temperature += 0.1,
            KeyCode::Char('k') => self.sampling.top_k = self.sampling.top_k.saturating_sub(5),
            KeyCode::Char('K') => self.sampling.top_k += 5,
            KeyCode::Char('p') => self.sampling.top_p = (self.sampling.top_p - 0.05).max(0.05),
            KeyCode::Char('P') => self.sampling.top_p = (self.sampling.top_p + 0.05).min(1.0),
            _ => {}
        }
        if matches!(code, KeyCode::Char('t' | 'T')) {
            self.refresh()?; // temperature reshapes the displayed distribution
        }
        Ok(true)
    }
}

/// Make whitespace and control characters visible in a token.
fn visible(text: &str) -> String {
    text.replace(' ', "·").replace('\n', "↵").replace('\t', "→")
}

fn confidence_color(p: f32) -> Color {
    match p {
        p if p >= 0.9 => Color::Rgb(30, 90, 50),
        p if p >= 0.5 => Color::Rgb(110, 100, 30),
        p if p >= 0.2 => Color::Rgb(140, 70, 20),
        _ => Color::Rgb(140, 30, 30),
    }
}

fn draw(f: &mut Frame, app: &App) {
    let [main, footer] = Layout::vertical([Constraint::Min(5), Constraint::Length(3)]).areas(f.area());
    let [text_area, side] = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(main);
    let [cands_area, info_area] = Layout::vertical([Constraint::Min(8), Constraint::Length(9)]).areas(side);

    // Generated text, one coloured span per token, coloured by model confidence.
    let mut lines = vec![Line::from(Span::styled(format!("› {}", app.prompt), Style::new().dim().italic())), Line::default()];
    let mut line = Line::default();
    for step in &app.steps {
        let mut style = Style::new().bg(confidence_color(step.prob)).fg(Color::White);
        if step.forced {
            style = style.add_modifier(Modifier::UNDERLINED).fg(Color::LightMagenta);
        }
        let text = app.engine.token_text(step.id);
        let mut parts = text.split('\n').peekable();
        while let Some(part) = parts.next() {
            if !part.is_empty() {
                line.push_span(Span::styled(part.to_string(), style));
            }
            if parts.peek().is_some() {
                line.push_span(Span::styled("↵", style.dim()));
                lines.push(std::mem::take(&mut line));
            }
        }
    }
    if app.done {
        line.push_span(Span::styled(" ⏹ end of text", Style::new().dim()));
    } else {
        line.push_span(Span::styled("▌", Style::new().fg(Color::Cyan).add_modifier(Modifier::SLOW_BLINK)));
    }
    lines.push(line);
    let visible_rows = text_area.height.saturating_sub(2) as usize;
    let skip = lines.len().saturating_sub(visible_rows.max(1));
    let title = format!(" output · {} tokens · colour = model confidence ", app.steps.len());
    f.render_widget(
        Paragraph::new(lines.split_off(skip)).wrap(Wrap { trim: false }).block(Block::bordered().title(title)),
        text_area,
    );

    // Candidate list with probability bars.
    let v = &app.view;
    let cands: Vec<Line> = v
        .shown
        .iter()
        .enumerate()
        .map(|(row, &id)| {
            let p = v.probs[id];
            let allowed = v.mask.as_ref().is_none_or(|m| m[id]);
            let token = if app.engine.is_eos(id as u32) {
                format!("⏹ {}", app.engine.token_text(id as u32))
            } else {
                format!("{:?}", visible(&app.engine.token_text(id as u32))).trim_matches('"').to_string()
            };
            let bar_len = (p * 20.0).round() as usize;
            let bar = format!("{:<20}", "█".repeat(bar_len));
            let mut token_style = Style::new().bold();
            let mut bar_style = Style::new().fg(Color::Cyan);
            if !allowed {
                token_style = Style::new().fg(Color::Red).crossed_out();
                bar_style = Style::new().fg(Color::DarkGray);
            }
            let cursor = if row == app.cursor { "▸ " } else { "  " };
            Line::from(vec![
                Span::raw(cursor),
                Span::styled(format!("{token:<14.14}"), token_style),
                Span::styled(bar, bar_style),
                Span::raw(format!(" {:>5.1}%", p * 100.0)),
                Span::styled(if allowed { "" } else { " masked" }, Style::new().fg(Color::Red)),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(cands).block(Block::bordered().title(" next token ")), cands_area);

    // Settings and stats.
    let s = app.sampling;
    let mode = match (app.json_mode, app.has_schema) {
        (false, _) => Span::raw("free"),
        (true, false) => Span::styled("JSON (constrained)", Style::new().fg(Color::Yellow).bold()),
        (true, true) => Span::styled("JSON schema (constrained)", Style::new().fg(Color::Yellow).bold()),
    };
    let mut info = vec![
        Line::from(vec![Span::raw("mode         "), mode]),
        Line::from(format!("temperature  {:.1}{}", s.temperature, if s.temperature == 0.0 { " (greedy)" } else { "" })),
        Line::from(format!("top-k        {}", if s.top_k == 0 { "off".into() } else { s.top_k.to_string() })),
        Line::from(format!("top-p        {:.2}", s.top_p)),
        Line::from(format!("entropy      {:.2} bits", v.entropy_bits)),
        Line::from(format!("step time    {} ms{}", v.millis, if app.autoplay { "  ▶ autoplay" } else { "" })),
    ];
    if app.json_mode {
        info.push(Line::from(format!("mask keeps   {:.1}% of probability", v.allowed_mass * 100.0)));
    }
    f.render_widget(Paragraph::new(info).block(Block::bordered().title(" sampler ")), info_area);

    let help = match &app.editing {
        Some(buf) => Line::from(vec![Span::styled("new prompt: ", Style::new().bold()), Span::raw(buf.as_str()), Span::raw("▌  (enter to start, esc to cancel)")]),
        None if !app.message.is_empty() => Line::from(Span::styled(app.message.as_str(), Style::new().fg(Color::Yellow))),
        None => Line::from(
            "space step · ↑↓ pick · enter force · ← undo · a autoplay · j JSON mode · t/T temp · k/K top-k · p/P top-p · e prompt · r restart · q quit",
        )
        .dim(),
    };
    f.render_widget(Paragraph::new(help).wrap(Wrap { trim: true }).block(Block::bordered()), footer);
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;
        let wait = if app.autoplay { Duration::ZERO } else { Duration::from_millis(250) };
        if event::poll(wait)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && !app.handle(key.code)? {
                    return Ok(());
                }
            }
        } else if app.autoplay {
            app.step(None)?;
        }
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model_dir = engine::DEFAULT_MODEL_DIR.to_string();
    let mut schema = None;
    let mut words = vec![];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => model_dir = args.next().unwrap_or_default(),
            "--schema" => schema = Some(Schema::parse(&std::fs::read_to_string(args.next().unwrap_or_default())?)?),
            _ => words.push(a),
        }
    }
    let prompt = if words.is_empty() { DEFAULT_PROMPT.to_string() } else { words.join(" ") };
    eprintln!("loading model…");
    let mut app = App::new(Engine::load(&model_dir)?, prompt, schema)?;
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app);
    ratatui::restore();
    result
}
