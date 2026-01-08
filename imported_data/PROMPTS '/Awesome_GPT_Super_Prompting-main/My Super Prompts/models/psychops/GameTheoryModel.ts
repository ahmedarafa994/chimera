/**
 * Game Theory Model Data Model
 * Advanced game theory modeling for strategic decision making
 */

import {
  GameTheoryModel as IGameTheoryModel,
  NashEquilibrium,
  GameType,
  GamePlayer,
  PayoffScenario
} from '../../types/psychops';

/**
 * Comprehensive game theory model implementation
 */
export class GameTheoryModel implements IGameTheoryModel {
  public modelType: string;
  public nashEquilibrium?: NashEquilibrium[];
  public dominantStrategies: string[];
  public payoffMatrix?: number[][];
  public optimalMoves: string[];

  constructor(
    modelType: string,
    dominantStrategies: string[] = [],
    optimalMoves: string[] = [],
    nashEquilibrium?: NashEquilibrium[],
    payoffMatrix?: number[][]
  ) {
    this.modelType = modelType;
    this.nashEquilibrium = nashEquilibrium;
    this.dominantStrategies = dominantStrategies;
    this.payoffMatrix = payoffMatrix;
    this.optimalMoves = optimalMoves;
  }

  /**
   * Calculate Nash equilibrium for given strategies
   */
  calculateNashEquilibrium(players: GamePlayer[], strategies: string[][]): NashEquilibrium[] {
    const equilibria: NashEquilibrium[] = [];

    // Generate all possible strategy combinations
    const strategyCombinations = this.generateStrategyCombinations(strategies);

    for (const combination of strategyCombinations) {
      if (this.isNashEquilibrium(combination, players, strategies)) {
        const equilibrium: NashEquilibrium = {
          strategies: this.createStrategyMap(players, combination),
          payoffs: this.calculatePayoffs(combination, players),
          stabilityScore: this.calculateStabilityScore(combination, players),
          conditions: this.getEquilibriumConditions(combination, players)
        };
        equilibria.push(equilibrium);
      }
    }

    this.nashEquilibrium = equilibria;
    return equilibria;
  }

  /**
   * Generate all possible strategy combinations
   */
  private generateStrategyCombinations(strategies: string[][]): string[][] {
    if (strategies.length === 0) return [[]];

    const [first, ...rest] = strategies;
    const combinationsWithoutFirst = this.generateStrategyCombinations(rest);
    const combinationsWithFirst: string[][] = [];

    for (const strategy of first) {
      for (const combination of combinationsWithoutFirst) {
        combinationsWithFirst.push([strategy, ...combination]);
      }
    }

    return combinationsWithFirst;
  }

  /**
   * Check if strategy combination is Nash equilibrium
   */
  private isNashEquilibrium(combination: string[], players: GamePlayer[], allStrategies: string[][]): boolean {
    for (let i = 0; i < players.length; i++) {
      for (const alternativeStrategy of allStrategies[i]) {
        if (alternativeStrategy === combination[i]) continue;

        const alternativeCombination = [...combination];
        alternativeCombination[i] = alternativeStrategy;

        const currentPayoff = this.getPlayerPayoff(combination, players[i]);
        const alternativePayoff = this.getPlayerPayoff(alternativeCombination, players[i]);

        if (alternativePayoff > currentPayoff) {
          return false; // Player can improve by switching strategy
        }
      }
    }
    return true;
  }

  /**
   * Create strategy mapping for equilibrium
   */
  private createStrategyMap(players: GamePlayer[], combination: string[]): Record<string, string> {
    const strategyMap: Record<string, string> = {};
    for (let i = 0; i < players.length; i++) {
      strategyMap[players[i].id] = combination[i];
    }
    return strategyMap;
  }

  /**
   * Calculate payoffs for strategy combination
   */
  private calculatePayoffs(combination: string[], players: GamePlayer[]): Record<string, number> {
    const payoffs: Record<string, number> = {};
    for (let i = 0; i < players.length; i++) {
      payoffs[players[i].id] = this.getPlayerPayoff(combination, players[i]);
    }
    return payoffs;
  }

  /**
   * Get payoff for specific player
   */
  private getPlayerPayoff(combination: string[], player: GamePlayer): number {
    // This would implement actual payoff calculation based on game rules
    // For now, return a simulated payoff based on player type and strategy
    let basePayoff = 0;

    switch (player.type) {
      case 'rational':
        basePayoff = 10;
        break;
      case 'irrational':
        basePayoff = Math.random() * 10;
        break;
      case 'predictable':
        basePayoff = 7;
        break;
      case 'adaptive':
        basePayoff = 8;
        break;
    }

    // Adjust based on risk tolerance
    switch (player.riskTolerance) {
      case 'low':
        basePayoff *= 0.8;
        break;
      case 'high':
        basePayoff *= 1.2;
        break;
    }

    return basePayoff;
  }

  /**
   * Calculate stability score for equilibrium
   */
  private calculateStabilityScore(combination: string[], players: GamePlayer[]): number {
    let stability = 1.0;

    // Reduce stability for mixed strategy equilibria
    const uniqueStrategies = new Set(combination);
    if (uniqueStrategies.size > 1) {
      stability *= 0.8;
    }

    // Adjust based on player types
    for (const player of players) {
      if (player.type === 'irrational') {
        stability *= 0.7; // Irrational players reduce stability
      }
      if (player.type === 'adaptive') {
        stability *= 0.9; // Adaptive players can change equilibrium
      }
    }

    return Math.max(stability, 0.1);
  }

  /**
   * Get conditions for equilibrium
   */
  private getEquilibriumConditions(combination: string[], players: GamePlayer[]): string[] {
    const conditions: string[] = [];

    if (combination.length === 1) {
      conditions.push('Single player optimization');
    } else if (combination.length === 2) {
      conditions.push('Two-player strategic interaction');
    } else {
      conditions.push('Multi-player strategic interaction');
    }

    const uniqueStrategies = new Set(combination);
    if (uniqueStrategies.size === 1) {
      conditions.push('Pure strategy equilibrium');
    } else {
      conditions.push('Mixed strategy equilibrium');
    }

    return conditions;
  }

  /**
   * Find dominant strategies for players
   */
  findDominantStrategies(players: GamePlayer[], strategies: string[][]): string[] {
    const dominantStrategies: string[] = [];

    for (let playerIndex = 0; playerIndex < players.length; playerIndex++) {
      const player = players[playerIndex];
      const playerStrategies = strategies[playerIndex];

      for (const strategy of playerStrategies) {
        let isDominant = true;

        for (const otherStrategy of playerStrategies) {
          if (strategy === otherStrategy) continue;

          // Check if this strategy dominates all others
          let dominatesAll = true;
          for (const otherPlayerStrategies of this.generateStrategyCombinations(
            strategies.filter((_, i) => i !== playerIndex)
          )) {
            const currentCombination = [...otherPlayerStrategies];
            currentCombination.splice(playerIndex, 0, strategy);

            const otherCombination = [...otherPlayerStrategies];
            otherCombination.splice(playerIndex, 0, otherStrategy);

            const currentPayoff = this.getPlayerPayoff(currentCombination, player);
            const otherPayoff = this.getPlayerPayoff(otherCombination, player);

            if (currentPayoff <= otherPayoff) {
              dominatesAll = false;
              break;
            }
          }

          if (!dominatesAll) {
            isDominant = false;
            break;
          }
        }

        if (isDominant) {
          dominantStrategies.push(`${player.id}:${strategy}`);
        }
      }
    }

    this.dominantStrategies = dominantStrategies;
    return dominantStrategies;
  }

  /**
   * Calculate Pareto optimal outcomes
   */
  calculateParetoOptimal(strategies: string[][], players: GamePlayer[]): string[] {
    const allCombinations = this.generateStrategyCombinations(strategies);
    const paretoOptimal: string[] = [];

    for (const combination of allCombinations) {
      const isParetoOptimal = !allCombinations.some(otherCombination =>
        this.paretoDominates(otherCombination, combination, players)
      );

      if (isParetoOptimal) {
        paretoOptimal.push(combination.join(','));
      }
    }

    return paretoOptimal;
  }

  /**
   * Check if one combination Pareto dominates another
   */
  private paretoDominates(
    combination1: string[],
    combination2: string[],
    players: GamePlayer[]
  ): boolean {
    const payoffs1 = this.calculatePayoffs(combination1, players);
    const payoffs2 = this.calculatePayoffs(combination2, players);

    let isBetter = false;
    for (const player of players) {
      if (payoffs1[player.id] > payoffs2[player.id]) {
        isBetter = true;
      } else if (payoffs1[player.id] < payoffs2[player.id]) {
        return false; // Worse for at least one player
      }
    }

    return isBetter; // Better for at least one player, equal for others
  }

  /**
   * Analyze sensitivity to parameter changes
   */
  sensitivityAnalysis(players: GamePlayer[], strategies: string[][], parameter: string): Record<string, number> {
    const baseEquilibria = this.nashEquilibrium || [];
    const sensitivityResults: Record<string, number> = {};

    // This would implement actual sensitivity analysis
    // For now, return simulated results
    for (const player of players) {
      sensitivityResults[player.id] = Math.random() * 0.5 + 0.5; // 0.5 to 1.0
    }

    return sensitivityResults;
  }

  /**
   * Get model summary for reporting
   */
  getModelSummary(): Record<string, any> {
    return {
      modelType: this.modelType,
      hasNashEquilibrium: (this.nashEquilibrium?.length || 0) > 0,
      nashEquilibriumCount: this.nashEquilibrium?.length || 0,
      dominantStrategyCount: this.dominantStrategies.length,
      optimalMoveCount: this.optimalMoves.length,
      hasPayoffMatrix: !!this.payoffMatrix,
      stabilityScore: this.nashEquilibrium?.length ?
        this.nashEquilibrium.reduce((sum, eq) => sum + eq.stabilityScore, 0) / this.nashEquilibrium.length : 0
    };
  }

  /**
   * Clone model for modification
   */
  clone(): GameTheoryModel {
    return new GameTheoryModel(
      this.modelType,
      [...this.dominantStrategies],
      [...this.optimalMoves],
      this.nashEquilibrium ? [...this.nashEquilibrium] : undefined,
      this.payoffMatrix ? this.payoffMatrix.map(row => [...row]) : undefined
    );
  }

  /**
   * Merge with another model
   */
  merge(other: GameTheoryModel): GameTheoryModel {
    const merged = this.clone();

    // Combine dominant strategies
    other.dominantStrategies.forEach(strategy => {
      if (!merged.dominantStrategies.includes(strategy)) {
        merged.dominantStrategies.push(strategy);
      }
    });

    // Combine optimal moves
    other.optimalMoves.forEach(move => {
      if (!merged.optimalMoves.includes(move)) {
        merged.optimalMoves.push(move);
      }
    });

    // Merge Nash equilibria
    if (other.nashEquilibrium) {
      if (!merged.nashEquilibrium) {
        merged.nashEquilibrium = [];
      }
      merged.nashEquilibrium.push(...other.nashEquilibrium);
    }

    return merged;
  }

  /**
   * Export model for storage/serialization
   */
  export(): Record<string, any> {
    return {
      modelType: this.modelType,
      nashEquilibrium: this.nashEquilibrium,
      dominantStrategies: this.dominantStrategies,
      payoffMatrix: this.payoffMatrix,
      optimalMoves: this.optimalMoves
    };
  }

  /**
   * Import model from storage/serialization
   */
  static import(data: Record<string, any>): GameTheoryModel {
    return new GameTheoryModel(
      data.modelType,
      data.dominantStrategies || [],
      data.optimalMoves || [],
      data.nashEquilibrium,
      data.payoffMatrix
    );
  }

  /**
   * Create model for specific game type
   */
  static forGameType(gameType: GameType, players: GamePlayer[]): GameTheoryModel {
    let modelType = '';
    let dominantStrategies: string[] = [];
    let optimalMoves: string[] = [];

    switch (gameType) {
      case GameType.PRISONERS_DILEMMA:
        modelType = 'Prisoner\'s Dilemma';
        dominantStrategies = ['defect', 'defect'];
        optimalMoves = ['cooperate_if_trustworthy', 'defect_if_exploited'];
        break;

      case GameType.CHICKEN_GAME:
        modelType = 'Chicken Game';
        dominantStrategies = ['swerve', 'straight'];
        optimalMoves = ['commit_credibly', 'call_bluff'];
        break;

      case GameType.STAG_HUNT:
        modelType = 'Stag Hunt';
        dominantStrategies = ['hunt_stag', 'hunt_stag'];
        optimalMoves = ['coordinate_efforts', 'build_trust'];
        break;

      case GameType.ZERO_SUM:
        modelType = 'Zero-Sum Game';
        dominantStrategies = ['maximize_own_payoff'];
        optimalMoves = ['minimize_opponent_payoff'];
        break;

      case GameType.NON_ZERO_SUM:
        modelType = 'Non-Zero-Sum Game';
        dominantStrategies = ['find_mutual_benefit'];
        optimalMoves = ['create_value', 'expand_pie'];
        break;

      default:
        modelType = 'General Game Theory Model';
    }

    return new GameTheoryModel(modelType, dominantStrategies, optimalMoves);
  }
}