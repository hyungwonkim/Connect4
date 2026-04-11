import type { Player } from "../engine/board";
import type { BasePlayer } from "../players/base";
import type { NNSession, NNTensorConstructor } from "../players/nn";
import { RandomPlayer } from "../players/random";
import { GreedyPlayer, EpsilonGreedyPlayer } from "../players/greedy";
import { DQNPlayer } from "../players/dqn";
import { PPOPlayer } from "../players/ppo";
import { AlphaZeroPlayer } from "../players/alphazero/player";

export type OpponentKind =
  | "human"
  | "random"
  | "greedy"
  | "epsilon_greedy"
  | "dqn"
  | "ppo"
  | "alphazero";

export interface OpponentConfig {
  kind: OpponentKind;
  alphaZeroSims?: number; // only for alphazero
}

export const OPPONENT_LABELS: Record<OpponentKind, string> = {
  human: "Human (2-player hotseat)",
  random: "Random",
  greedy: "Greedy (rule-based)",
  epsilon_greedy: "ε-Greedy",
  dqn: "DQN (RL)",
  ppo: "PPO (RL)",
  alphazero: "AlphaZero (MCTS + NN)",
};

export interface SessionLoader {
  load(modelName: "dqn" | "ppo" | "alphazero"): Promise<{
    session: NNSession;
    TensorCtor: NNTensorConstructor;
  }>;
}

/**
 * Create a player of the given kind, lazily loading ONNX sessions as needed.
 */
export async function makeOpponent(
  config: OpponentConfig,
  playerId: Player,
  loader: SessionLoader,
): Promise<BasePlayer | null> {
  switch (config.kind) {
    case "human":
      return null; // handled specially by the game loop
    case "random":
      return new RandomPlayer();
    case "greedy":
      return new GreedyPlayer(playerId);
    case "epsilon_greedy":
      return new EpsilonGreedyPlayer(playerId, 0.1);
    case "dqn": {
      const { session, TensorCtor } = await loader.load("dqn");
      return new DQNPlayer(playerId, session, TensorCtor);
    }
    case "ppo": {
      const { session, TensorCtor } = await loader.load("ppo");
      return new PPOPlayer(playerId, session, TensorCtor);
    }
    case "alphazero": {
      const { session, TensorCtor } = await loader.load("alphazero");
      return new AlphaZeroPlayer(playerId, session, TensorCtor, {
        numSimulations: config.alphaZeroSims ?? 200,
      });
    }
  }
}
