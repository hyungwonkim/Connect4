import "./style.css";
import { P1, P2, type Player } from "./engine/board";
import type { BasePlayer } from "./players/base";
import { BoardView, DEFAULT_COLORS, type ColorMap } from "./ui/view";
import { GameController } from "./ui/game";
import { OrtLoader } from "./ui/ort-loader";
import {
  OPPONENT_LABELS,
  makeOpponent,
  type OpponentConfig,
  type OpponentKind,
} from "./ui/opponents";

const app = document.getElementById("app")!;
const loader = new OrtLoader(import.meta.env.BASE_URL);

let currentController: GameController | null = null;

function renderStartScreen(): void {
  app.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "start-screen";

  const title = document.createElement("h1");
  title.textContent = "Connect 4";
  wrap.appendChild(title);

  const subtitle = document.createElement("p");
  subtitle.className = "subtitle";
  subtitle.textContent = "Pick an opponent. You play yellow.";
  wrap.appendChild(subtitle);

  const form = document.createElement("div");
  form.className = "opponent-picker";

  const opponentOrder: OpponentKind[] = [
    "human",
    "random",
    "greedy",
    "epsilon_greedy",
    "dqn",
    "ppo",
    "alphazero",
  ];

  let selected: OpponentKind = "alphazero";
  let alphaZeroSims = 200;
  let humanFirst = true;

  const buttons: HTMLButtonElement[] = [];
  for (const kind of opponentOrder) {
    const btn = document.createElement("button");
    btn.className = "opponent-btn";
    btn.textContent = OPPONENT_LABELS[kind];
    btn.dataset.kind = kind;
    btn.addEventListener("click", () => {
      selected = kind;
      buttons.forEach((b) =>
        b.classList.toggle("selected", b.dataset.kind === kind),
      );
      simsRow.style.display = kind === "alphazero" ? "flex" : "none";
      orderRow.style.display = kind === "human" ? "none" : "flex";
    });
    buttons.push(btn);
    form.appendChild(btn);
  }
  form.querySelector<HTMLButtonElement>(`[data-kind="${selected}"]`)?.classList.add("selected");
  wrap.appendChild(form);

  // Turn-order toggle: who goes first? (Hidden in human-vs-human mode.)
  const orderRow = document.createElement("div");
  orderRow.className = "order-row";
  const orderLabel = document.createElement("label");
  orderLabel.textContent = "You play:";
  orderRow.appendChild(orderLabel);
  const orderBtns = document.createElement("div");
  orderBtns.className = "order-btns";
  const firstBtn = document.createElement("button");
  firstBtn.className = "order-btn selected";
  firstBtn.textContent = "First (yellow)";
  const secondBtn = document.createElement("button");
  secondBtn.className = "order-btn";
  secondBtn.textContent = "Second (yellow)";
  firstBtn.addEventListener("click", () => {
    humanFirst = true;
    firstBtn.classList.add("selected");
    secondBtn.classList.remove("selected");
  });
  secondBtn.addEventListener("click", () => {
    humanFirst = false;
    secondBtn.classList.add("selected");
    firstBtn.classList.remove("selected");
  });
  orderBtns.appendChild(firstBtn);
  orderBtns.appendChild(secondBtn);
  orderRow.appendChild(orderBtns);
  wrap.appendChild(orderRow);

  // AlphaZero difficulty
  const simsRow = document.createElement("div");
  simsRow.className = "sims-row";
  const simsLabel = document.createElement("label");
  simsLabel.textContent = "AlphaZero strength:";
  simsRow.appendChild(simsLabel);
  const simsSelect = document.createElement("select");
  for (const [label, value] of [
    ["Easy (100 sims)", 100],
    ["Medium (200 sims)", 200],
    ["Hard (400 sims)", 400],
    ["Expert (800 sims)", 800],
  ] as const) {
    const opt = document.createElement("option");
    opt.value = String(value);
    opt.textContent = label;
    if (value === alphaZeroSims) opt.selected = true;
    simsSelect.appendChild(opt);
  }
  simsSelect.addEventListener("change", () => {
    alphaZeroSims = parseInt(simsSelect.value, 10);
  });
  simsRow.appendChild(simsSelect);
  wrap.appendChild(simsRow);

  const startBtn = document.createElement("button");
  startBtn.className = "start-btn";
  startBtn.textContent = "Start game";
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    startBtn.textContent = "Loading…";
    try {
      await startGame({ kind: selected, alphaZeroSims }, humanFirst);
    } catch (err) {
      console.error(err);
      startBtn.textContent = "Failed — try again";
      startBtn.disabled = false;
    }
  });
  wrap.appendChild(startBtn);

  app.appendChild(wrap);
}

async function startGame(config: OpponentConfig, humanFirst: boolean): Promise<void> {
  app.innerHTML = "";
  const gameRoot = document.createElement("div");
  gameRoot.className = "game-root";
  app.appendChild(gameRoot);

  // In human-vs-human the order toggle is hidden; use default colors.
  // Otherwise the human always renders as yellow regardless of P1/P2.
  const humanId: Player = humanFirst ? P1 : P2;
  const opponentId: Player = humanFirst ? P2 : P1;
  const colors: ColorMap =
    config.kind === "human"
      ? DEFAULT_COLORS
      : ({ [humanId]: "yellow", [opponentId]: "red" } as ColorMap);

  const view = new BoardView(gameRoot, colors);
  view.setStatus("Loading opponent…", "thinking");

  const opponent: BasePlayer | null =
    config.kind === "human" ? null : await makeOpponent(config, opponentId, loader);

  const players: { 1: BasePlayer | null; 2: BasePlayer | null } =
    config.kind === "human"
      ? { [P1]: null, [P2]: null }
      : humanFirst
        ? { [P1]: null, [P2]: opponent }
        : { [P1]: opponent, [P2]: null };

  const controller = new GameController(view, players);
  currentController = controller;

  view.setOnRestart(() => {
    controller.cancel();
    renderStartScreen();
  });

  await controller.run();
}

renderStartScreen();

// Avoid the "declared but unused" TS complaint if hot-reload clears it.
void currentController;
