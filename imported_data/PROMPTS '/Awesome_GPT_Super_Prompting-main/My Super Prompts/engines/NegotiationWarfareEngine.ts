/**
 * Negotiation Warfare Engine for PSYCH OPS
 * Advanced strategic negotiation and psychological warfare implementation
 */

import { EventEmitter } from 'events';
import {
  NegotiationWarfareRequest,
  NegotiationWarfareResponse,
  NegotiationParty,
  PowerDynamic,
  StrategicAnalysis,
  TacticalRecommendation,
  GameTheoryModel,
  NegotiationScript,
  NegotiationPhase,
  CommunicationStrategy,
  ConcessionStrategy,
  RiskAssessment,
  PsychologicalProfile,
  GameType
} from '../types/psychops';

/**
 * Advanced Negotiation Warfare Engine
 */
export class NegotiationWarfareEngine extends EventEmitter {
  private strategyDatabase = new Map<string, StrategicAnalysis>();
  private tacticLibrary = new Map<string, TacticalRecommendation>();
  private gameTheoryModels = new Map<GameType, GameTheoryModel>();

  constructor() {
    super();
    this.initializeEngine();
  }

  /**
   * Initialize the negotiation warfare engine
   */
  private async initializeEngine(): Promise<void> {
    try {
      await this.loadStrategyDatabase();
      await this.loadTacticLibrary();
      await this.initializeGameTheoryModels();

      this.emit('engineInitialized', {
        strategies: this.strategyDatabase.size,
        tactics: this.tacticLibrary.size,
        models: this.gameTheoryModels.size
      });
    } catch (error) {
      this.emit('engineError', { error: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  /**
   * Load strategy database
   */
  private async loadStrategyDatabase(): Promise<void> {
    const strategies: StrategicAnalysis[] = [
      {
        overallStrategy: 'Dominance Through Information Asymmetry',
        keyObjectives: [
          'Control information flow',
          'Create dependency on your knowledge',
          'Exploit information gaps'
        ],
        strategicAdvantages: [
          'Superior intelligence gathering',
          'Psychological profiling capabilities',
          'Real-time adaptation'
        ],
        strategicDisadvantages: [
          'Resource intensive',
          'Detection risk',
          'Ethical considerations'
        ],
        competitiveAnalysis: [
          'Target has limited counter-intelligence',
          'Psychological vulnerabilities identified',
          'Negotiation leverage established'
        ]
      },
      {
        overallStrategy: 'Psychological Manipulation Framework',
        keyObjectives: [
          'Exploit cognitive biases',
          'Control emotional states',
          'Shape decision frameworks'
        ],
        strategicAdvantages: [
          'Deep psychological insights',
          'Behavioral prediction models',
          'Emotional manipulation techniques'
        ],
        strategicDisadvantages: [
          'Backlash potential',
          'Relationship damage',
          'Long-term consequences'
        ],
        competitiveAnalysis: [
          'Target emotional vulnerabilities mapped',
          'Cognitive bias exploitation opportunities',
          'Behavioral conditioning possibilities'
        ]
      },
      {
        overallStrategy: 'Game Theory Optimization',
        keyObjectives: [
          'Model strategic interactions',
          'Predict opponent moves',
          'Optimize payoff structures'
        ],
        strategicAdvantages: [
          'Mathematical modeling capabilities',
          'Strategic equilibrium analysis',
          'Payoff optimization'
        ],
        strategicDisadvantages: [
          'Computational complexity',
          'Assumption dependencies',
          'Real-time adaptation challenges'
        ],
        competitiveAnalysis: [
          'Multiple game theory models available',
          'Real-time strategy adjustment',
          'Equilibrium exploitation capabilities'
        ]
      }
    ];

    for (const strategy of strategies) {
      this.strategyDatabase.set(strategy.overallStrategy, strategy);
    }

    this.emit('strategyDatabaseLoaded', { count: strategies.length });
  }

  /**
   * Load tactic library
   */
  private async loadTacticLibrary(): Promise<void> {
    const tactics: TacticalRecommendation[] = [
      {
        name: 'Anchoring Manipulation',
        description: 'Set extreme initial positions to shift negotiation range',
        timing: 'Opening phase',
        expectedImpact: 0.8,
        resourceRequirements: [
          'Psychological research on anchoring effects',
          'Preparation of extreme positions',
          'Timing control mechanisms'
        ]
      },
      {
        name: 'Reciprocity Engineering',
        description: 'Create artificial obligation through strategic concessions',
        timing: 'Middle phase',
        expectedImpact: 0.7,
        resourceRequirements: [
          'Valuable concession items',
          'Concession timing control',
          'Reciprocity tracking systems'
        ]
      },
      {
        name: 'Deadline Pressure',
        description: 'Manufacture time pressure to force suboptimal decisions',
        timing: 'Closing phase',
        expectedImpact: 0.9,
        resourceRequirements: [
          'Deadline creation mechanisms',
          'Alternative option preparation',
          'Pressure monitoring systems'
        ]
      },
      {
        name: 'Good Cop/Bad Cop',
        description: 'Use multiple personas to manipulate emotional responses',
        timing: 'Throughout negotiation',
        expectedImpact: 0.75,
        resourceRequirements: [
          'Multiple negotiator coordination',
          'Persona consistency maintenance',
          'Emotional response tracking'
        ]
      },
      {
        name: 'Information Asymmetry Exploitation',
        description: 'Use superior knowledge to manipulate perceptions',
        timing: 'Information exchange phases',
        expectedImpact: 0.85,
        resourceRequirements: [
          'Intelligence gathering systems',
          'Information validation',
          'Asymmetry maintenance'
        ]
      }
    ];

    for (const tactic of tactics) {
      this.tacticLibrary.set(tactic.name, tactic);
    }

    this.emit('tacticLibraryLoaded', { count: tactics.length });
  }

  /**
   * Initialize game theory models
   */
  private async initializeGameTheoryModels(): Promise<void> {
    const gameTypes = Object.values(GameType);

    for (const gameType of gameTypes) {
      const model = GameTheoryModel.forGameType(gameType, []);
      this.gameTheoryModels.set(gameType, model);
    }

    this.emit('gameTheoryModelsInitialized', { count: gameTypes.length });
  }

  /**
   * Execute negotiation warfare
   */
  async executeNegotiationWarfare(request: NegotiationWarfareRequest): Promise<NegotiationWarfareResponse> {
    try {
      this.emit('negotiationWarfareStarted', {
        context: request.context,
        partyCount: request.parties.length,
        outcomeCount: request.desiredOutcomes.length
      });

      // Analyze strategic landscape
      const strategicAnalysis = await this.analyzeStrategicLandscape(request);

      // Generate tactical recommendations
      const tacticalRecommendations = await this.generateTacticalRecommendations(request);

      // Create game theory model
      const gameTheoryModel = await this.createGameTheoryModel(request);

      // Generate negotiation script
      const negotiationScript = await this.generateNegotiationScript(request, tacticalRecommendations);

      // Calculate success probability
      const successProbability = this.calculateSuccessProbability(request, strategicAnalysis);

      // Assess risks
      const riskAssessment = await this.assessNegotiationRisks(request, strategicAnalysis);

      const response: NegotiationWarfareResponse = {
        id: this.generateResponseId(),
        strategicAnalysis,
        tacticalRecommendations,
        gameTheoryModel,
        negotiationScript,
        successProbability,
        riskAssessment
      };

      this.emit('negotiationWarfareCompleted', {
        responseId: response.id,
        successProbability: response.successProbability,
        recommendationCount: response.tacticalRecommendations.length
      });

      return response;

    } catch (error) {
      this.emit('negotiationWarfareError', {
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Analyze strategic landscape
   */
  private async analyzeStrategicLandscape(request: NegotiationWarfareRequest): Promise<StrategicAnalysis> {
    // Analyze power dynamics
    const powerAnalysis = this.analyzePowerDynamics(request.parties);

    // Identify key objectives and constraints
    const objectiveAnalysis = this.analyzeObjectives(request);

    // Assess competitive advantages
    const competitiveAnalysis = this.assessCompetitiveAdvantages(request);

    // Select optimal strategy
    const optimalStrategy = this.selectOptimalStrategy(request, powerAnalysis);

    return {
      overallStrategy: optimalStrategy,
      keyObjectives: objectiveAnalysis.keyObjectives,
      strategicAdvantages: powerAnalysis.advantages,
      strategicDisadvantages: powerAnalysis.disadvantages,
      competitiveAnalysis
    };
  }

  /**
   * Analyze power dynamics between parties
   */
  private analyzePowerDynamics(parties: NegotiationParty[]): {
    advantages: string[];
    disadvantages: string[];
    powerBalances: Record<string, number>;
  } {
    const advantages: string[] = [];
    const disadvantages: string[] = [];
    const powerBalances: Record<string, number> = {};

    for (const party of parties) {
      let powerScore = 0.5; // Base power

      // Analyze power dynamics
      for (const dynamic of party.powerDynamics) {
        powerScore += dynamic.powerBalance;

        if (dynamic.influenceFactors.includes('information_advantage')) {
          advantages.push(`${party.name} has information advantage`);
        }
        if (dynamic.influenceFactors.includes('psychological_leverage')) {
          advantages.push(`${party.name} has psychological leverage`);
        }
      }

      // Adjust based on leverage points
      powerScore += party.leveragePoints.length * 0.1;

      // Adjust based on interests
      powerScore += party.interests.length * 0.05;

      powerBalances[party.id] = Math.min(powerScore, 1.0);
    }

    // Identify overall advantages and disadvantages
    const maxPower = Math.max(...Object.values(powerBalances));
    const minPower = Math.min(...Object.values(powerBalances));

    if (maxPower > 0.7) {
      advantages.push('Significant power advantage established');
    }
    if (minPower < 0.3) {
      disadvantages.push('Power disadvantage requires careful management');
    }

    return { advantages, disadvantages, powerBalances };
  }

  /**
   * Analyze negotiation objectives
   */
  private analyzeObjectives(request: NegotiationWarfareRequest): {
    keyObjectives: string[];
    priorityOrder: string[];
    flexibilityAssessment: Record<string, 'high' | 'medium' | 'low'>;
  } {
    const keyObjectives = [...request.desiredOutcomes];

    // Assess objective priorities based on context
    const priorityOrder = this.prioritizeObjectives(keyObjectives, request);

    // Assess flexibility for each objective
    const flexibilityAssessment: Record<string, 'high' | 'medium' | 'low'> = {};
    for (const objective of keyObjectives) {
      flexibilityAssessment[objective] = this.assessObjectiveFlexibility(objective, request);
    }

    return { keyObjectives, priorityOrder, flexibilityAssessment };
  }

  /**
   * Prioritize objectives based on context and constraints
   */
  private prioritizeObjectives(objectives: string[], request: NegotiationWarfareRequest): string[] {
    // Simple prioritization - in production, this would be more sophisticated
    return objectives.sort((a, b) => {
      // Prioritize based on available leverage
      const aLeverage = request.availableLeverage.some(lev => a.toLowerCase().includes(lev.toLowerCase())) ? 1 : 0;
      const bLeverage = request.availableLeverage.some(lev => b.toLowerCase().includes(lev.toLowerCase())) ? 1 : 0;

      if (aLeverage !== bLeverage) {
        return bLeverage - aLeverage;
      }

      // Prioritize based on timeline constraints
      if (request.timeline) {
        const daysUntilDeadline = (request.timeline.getTime() - Date.now()) / (1000 * 60 * 60 * 24);
        if (daysUntilDeadline < 7) {
          return 0; // Maintain original order under time pressure
        }
      }

      return 0;
    });
  }

  /**
   * Assess flexibility for specific objective
   */
  private assessObjectiveFlexibility(objective: string, request: NegotiationWarfareRequest): 'high' | 'medium' | 'low' {
    // Assess based on constraints and leverage
    if (request.constraints && request.constraints.some(constraint =>
      objective.toLowerCase().includes(constraint.toLowerCase())
    )) {
      return 'low';
    }

    if (request.availableLeverage.some(lev =>
      objective.toLowerCase().includes(lev.toLowerCase())
    )) {
      return 'high';
    }

    return 'medium';
  }

  /**
   * Assess competitive advantages
   */
  private assessCompetitiveAdvantages(request: NegotiationWarfareRequest): string[] {
    const advantages: string[] = [];

    // Analyze available leverage
    if (request.availableLeverage.length > 0) {
      advantages.push(`Strong leverage position with ${request.availableLeverage.length} leverage points`);
    }

    // Analyze party profiles
    for (const party of request.parties) {
      if (party.psychologicalProfile && party.psychologicalProfile.profileScore > 0.7) {
        advantages.push(`${party.name} psychological profile provides manipulation opportunities`);
      }

      if (party.leveragePoints.length > 2) {
        advantages.push(`${party.name} has multiple leverage points for exploitation`);
      }
    }

    // Analyze timeline advantage
    if (request.timeline) {
      const daysUntilDeadline = (request.timeline.getTime() - Date.now()) / (1000 * 60 * 60 * 24);
      if (daysUntilDeadline > 30) {
        advantages.push('Extended timeline allows for comprehensive strategy development');
      } else if (daysUntilDeadline < 7) {
        advantages.push('Compressed timeline creates pressure opportunities');
      }
    }

    return advantages;
  }

  /**
   * Select optimal strategy
   */
  private selectOptimalStrategy(request: NegotiationWarfareRequest, powerAnalysis: any): string {
    // Strategy selection logic based on context
    if (powerAnalysis.powerBalances && Object.values(powerAnalysis.powerBalances)[0] > 0.7) {
      return 'Dominance Through Information Asymmetry';
    }

    if (request.parties.some(party => party.psychologicalProfile && party.psychologicalProfile.profileScore > 0.6)) {
      return 'Psychological Manipulation Framework';
    }

    return 'Game Theory Optimization';
  }

  /**
   * Generate tactical recommendations
   */
  private async generateTacticalRecommendations(request: NegotiationWarfareRequest): Promise<TacticalRecommendation[]> {
    const recommendations: TacticalRecommendation[] = [];

    // Select tactics based on strategic analysis
    const availableTactics = Array.from(this.tacticLibrary.values());

    // Filter tactics based on context and constraints
    const applicableTactics = availableTactics.filter(tactic =>
      this.isTacticApplicable(tactic, request)
    );

    // Score and rank tactics
    const scoredTactics = applicableTactics.map(tactic => ({
      tactic,
      score: this.scoreTacticForContext(tactic, request)
    }));

    scoredTactics.sort((a, b) => b.score - a.score);

    // Return top tactics
    return scoredTactics.slice(0, 5).map(item => item.tactic);
  }

  /**
   * Check if tactic is applicable to current context
   */
  private isTacticApplicable(tactic: TacticalRecommendation, request: NegotiationWarfareRequest): boolean {
    // Check timing compatibility
    if (request.timeline) {
      const daysUntilDeadline = (request.timeline.getTime() - Date.now()) / (1000 * 60 * 60 * 24);
      if (daysUntilDeadline < 3 && tactic.timing === 'Opening phase') {
        return false; // Not enough time for opening phase tactics
      }
    }

    // Check resource availability
    if (tactic.resourceRequirements.some(req =>
      !this.isResourceAvailable(req, request)
    )) {
      return false;
    }

    return true;
  }

  /**
   * Check if specific resource is available
   */
  private isResourceAvailable(resource: string, request: NegotiationWarfareRequest): boolean {
    // Simple resource availability check
    const resourceKeywords = resource.toLowerCase().split(' ');

    return request.availableLeverage.some(lev =>
      resourceKeywords.some(keyword => lev.toLowerCase().includes(keyword))
    );
  }

  /**
   * Score tactic for current context
   */
  private scoreTacticForContext(tactic: TacticalRecommendation, request: NegotiationWarfareRequest): number {
    let score = tactic.expectedImpact;

    // Adjust based on risk tolerance
    if (request.riskTolerance === 'low' && tactic.expectedImpact > 0.8) {
      score -= 0.2; // Reduce score for high-impact tactics under low risk tolerance
    }

    if (request.riskTolerance === 'high' && tactic.expectedImpact < 0.6) {
      score -= 0.1; // Reduce score for low-impact tactics under high risk tolerance
    }

    // Adjust based on available leverage
    const leverageMatch = tactic.resourceRequirements.some(req =>
      request.availableLeverage.some(lev =>
        req.toLowerCase().includes(lev.toLowerCase()) ||
        lev.toLowerCase().includes(req.toLowerCase())
      )
    );

    if (leverageMatch) {
      score += 0.2;
    }

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Create game theory model for negotiation
   */
  private async createGameTheoryModel(request: NegotiationWarfareRequest): Promise<GameTheoryModel> {
    // Determine appropriate game type
    const gameType = this.determineGameType(request);

    // Create player models
    const players = request.parties.map(party => ({
      id: party.id,
      name: party.name,
      type: this.determinePlayerType(party),
      strategyPreferences: party.leveragePoints,
      riskTolerance: request.riskTolerance,
      informationAccess: party.interests
    }));

    // Generate strategies for each player
    const strategies = players.map(player =>
      this.generatePlayerStrategies(player, request)
    );

    // Create payoff scenarios
    const payoffScenarios = this.generatePayoffScenarios(request, players);

    const model = new GameTheoryModel(
      `Negotiation Model - ${gameType}`,
      [], // Will be filled by calculateNashEquilibrium
      []  // Will be filled by findDominantStrategies
    );

    // Calculate equilibria and dominant strategies
    model.calculateNashEquilibrium(players, strategies);
    model.findDominantStrategies(players, strategies);

    return model;
  }

  /**
   * Determine appropriate game type for negotiation
   */
  private determineGameType(request: NegotiationWarfareRequest): GameType {
    if (request.parties.length === 2) {
      if (request.desiredOutcomes.some(outcome =>
        outcome.toLowerCase().includes('compete') || outcome.toLowerCase().includes('conflict')
      )) {
        return GameType.CHICKEN_GAME;
      }
      if (request.desiredOutcomes.some(outcome =>
        outcome.toLowerCase().includes('cooperate') || outcome.toLowerCase().includes('mutual')
      )) {
        return GameType.STAG_HUNT;
      }
      return GameType.PRISONERS_DILEMMA;
    }

    if (request.desiredOutcomes.length === 1) {
      return GameType.ZERO_SUM;
    }

    return GameType.NON_ZERO_SUM;
  }

  /**
   * Determine player type based on profile
   */
  private determinePlayerType(party: NegotiationParty): 'rational' | 'irrational' | 'predictable' | 'adaptive' {
    if (party.psychologicalProfile) {
      if (party.psychologicalProfile.profileScore > 0.8) {
        return 'predictable';
      }
      if (party.psychologicalProfile.personalityTraits.some(trait =>
        trait.name.toLowerCase().includes('neurotic') || trait.name.toLowerCase().includes('impulsive')
      )) {
        return 'irrational';
      }
      return 'adaptive';
    }

    return 'rational';
  }

  /**
   * Generate strategies for specific player
   */
  private generatePlayerStrategies(player: any, request: NegotiationWarfareRequest): string[] {
    const strategies = ['cooperate', 'compete', 'compromise'];

    // Add player-specific strategies based on leverage points
    strategies.push(...player.strategyPreferences.map((pref: string) => `leverage_${pref}`));

    // Add risk-based strategies
    if (player.riskTolerance === 'high') {
      strategies.push('aggressive', 'high_stakes');
    } else if (player.riskTolerance === 'low') {
      strategies.push('conservative', 'risk_averse');
    }

    return strategies;
  }

  /**
   * Generate payoff scenarios
   */
  private generatePayoffScenarios(request: NegotiationWarfareRequest, players: any[]): any[] {
    const scenarios = [];

    // Generate scenarios based on desired outcomes
    for (const outcome of request.desiredOutcomes) {
      const payoffs: Record<string, number> = {};

      for (const player of players) {
        if (outcome.toLowerCase().includes(player.name.toLowerCase())) {
          payoffs[player.id] = 10; // High payoff for achieving desired outcome
        } else {
          payoffs[player.id] = 5; // Moderate payoff for partial success
        }
      }

      scenarios.push({
        name: outcome,
        payoffs,
        probability: 1 / request.desiredOutcomes.length,
        conditions: [`Outcome: ${outcome}`]
      });
    }

    return scenarios;
  }

  /**
   * Generate negotiation script
   */
  private async generateNegotiationScript(
    request: NegotiationWarfareRequest,
    tactics: TacticalRecommendation[]
  ): Promise<NegotiationScript> {
    const phases = this.generateNegotiationPhases(request, tactics);
    const communicationStrategies = this.generateCommunicationStrategies(request);
    const concessionStrategies = this.generateConcessionStrategies(request);
    const closingStrategies = this.generateClosingStrategies(request);

    return {
      phases,
      communicationStrategies,
      concessionStrategies,
      closingStrategies
    };
  }

  /**
   * Generate negotiation phases
   */
  private generateNegotiationPhases(request: NegotiationWarfareRequest, tactics: TacticalRecommendation[]): NegotiationPhase[] {
    const phases: NegotiationPhase[] = [];

    // Opening phase
    phases.push({
      name: 'Strategic Opening',
      objectives: [
        'Establish psychological dominance',
        'Set negotiation frame',
        'Gather initial intelligence'
      ],
      keyMessages: [
        'Define negotiation boundaries',
        'Present initial position',
        'Test opponent responses'
      ],
      tacticalMoves: tactics
        .filter(t => t.timing === 'Opening phase')
        .map(t => t.name),
      transitionTriggers: [
        'Initial positions established',
        'Power dynamics clarified',
        'Information exchange initiated'
      ]
    });

    // Middle phase
    phases.push({
      name: 'Psychological Warfare',
      objectives: [
        'Exploit identified vulnerabilities',
        'Shape opponent perceptions',
        'Control information flow'
      ],
      keyMessages: [
        'Challenge opponent assumptions',
        'Present controlled information',
        'Manage emotional responses'
      ],
      tacticalMoves: tactics
        .filter(t => t.timing === 'Middle phase')
        .map(t => t.name),
      transitionTriggers: [
        'Key vulnerabilities exploited',
        'Opponent compliance increased',
        'Concession opportunities created'
      ]
    });

    // Closing phase
    phases.push({
      name: 'Strategic Closure',
      objectives: [
        'Secure desired outcomes',
        'Minimize concessions',
        'Establish control mechanisms'
      ],
      keyMessages: [
        'Present final position',
        'Create urgency for closure',
        'Document agreements'
      ],
      tacticalMoves: tactics
        .filter(t => t.timing === 'Closing phase')
        .map(t => t.name),
      transitionTriggers: [
        'Objectives achieved',
        'Opponent commitment secured',
        'Negotiation concluded'
      ]
    });

    return phases;
  }

  /**
   * Generate communication strategies
   */
  private generateCommunicationStrategies(request: NegotiationWarfareRequest): CommunicationStrategy[] {
    return [
      {
        type: 'Psychological Framing',
        style: 'Authoritative yet flexible',
        messageFraming: [
          'Emphasize mutual benefits',
          'Highlight potential losses',
          'Create aspiration for agreement'
        ],
        timingConsiderations: [
          'Match opponent communication style',
          'Use appropriate psychological techniques',
          'Maintain message consistency'
        ]
      },
      {
        type: 'Information Control',
        style: 'Selective disclosure',
        messageFraming: [
          'Reveal information strategically',
          'Control narrative development',
          'Manage perception of facts'
        ],
        timingConsiderations: [
          'Time disclosures for maximum impact',
          'Use information as leverage',
          'Maintain plausible deniability'
        ]
      }
    ];
  }

  /**
   * Generate concession strategies
   */
  private generateConcessionStrategies(request: NegotiationWarfareRequest): ConcessionStrategy[] {
    return [
      {
        type: 'Strategic Concession',
        value: 'Moderate value item',
        timing: 'After opponent resistance',
        strategicPurpose: 'Build reciprocity obligation'
      },
      {
        type: 'False Concession',
        value: 'Apparent major concession',
        timing: 'Critical negotiation moment',
        strategicPurpose: 'Extract significant counter-concession'
      },
      {
        type: 'Conditional Concession',
        value: 'Valuable but conditional item',
        timing: 'When leverage is strong',
        strategicPurpose: 'Reinforce power position'
      }
    ];
  }

  /**
   * Generate closing strategies
   */
  private generateClosingStrategies(request: NegotiationWarfareRequest): string[] {
    return [
      'Create artificial deadline pressure',
      'Present final offer as take-it-or-leave-it',
      'Emphasize costs of non-agreement',
      'Document all agreements clearly',
      'Establish monitoring mechanisms',
      'Plan for post-negotiation control'
    ];
  }

  /**
   * Calculate success probability
   */
  private calculateSuccessProbability(request: NegotiationWarfareRequest, strategicAnalysis: StrategicAnalysis): number {
    let probability = 0.5; // Base probability

    // Adjust based on strategic advantages
    probability += strategicAnalysis.strategicAdvantages.length * 0.1;

    // Adjust based on power dynamics
    const powerImbalance = Math.max(...Object.values(
      request.parties.reduce((acc, party) => {
        acc[party.id] = party.powerDynamics.reduce((sum, dynamic) => sum + dynamic.powerBalance, 0);
        return acc;
      }, {} as Record<string, number>)
    ));

    probability += powerImbalance * 0.2;

    // Adjust based on available leverage
    probability += request.availableLeverage.length * 0.05;

    // Adjust based on risk tolerance
    if (request.riskTolerance === 'high') {
      probability += 0.1;
    } else if (request.riskTolerance === 'low') {
      probability -= 0.1;
    }

    return Math.max(0.1, Math.min(0.95, probability));
  }

  /**
   * Assess negotiation risks
   */
  private async assessNegotiationRisks(request: NegotiationWarfareRequest, strategicAnalysis: StrategicAnalysis): Promise<RiskAssessment> {
    let overallRisk = 0;

    // Risk from strategic complexity
    overallRisk += strategicAnalysis.strategicAdvantages.length * 0.05;

    // Risk from opponent capabilities
    for (const party of request.parties) {
      if (party.psychologicalProfile && party.psychologicalProfile.profileScore > 0.7) {
        overallRisk += 0.1; // Opponent with good profile is riskier
      }
    }

    // Risk from time pressure
    if (request.timeline) {
      const daysUntilDeadline = (request.timeline.getTime() - Date.now()) / (1000 * 60 * 60 * 24);
      if (daysUntilDeadline < 7) {
        overallRisk += 0.2;
      }
    }

    return {
      overallRisk: Math.min(overallRisk, 1.0),
      riskCategories: {
        detection: overallRisk * 0.8,
        backlash: overallRisk * 0.6,
        failure: overallRisk * 0.7,
        ethical: overallRisk * 0.9
      },
      mitigationStrategies: [
        'Maintain operational security',
        'Have contingency plans',
        'Monitor for detection indicators',
        'Prepare plausible deniability'
      ],
      worstCaseScenarios: [
        'Complete negotiation failure',
        'Opponent counter-manipulation',
        'Reputation damage',
        'Legal consequences'
      ],
      detectionProbability: overallRisk * 0.6
    };
  }

  /**
   * Generate unique response ID
   */
  private generateResponseId(): string {
    return `negotiation_warfare_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get available strategies
   */
  getAvailableStrategies(): StrategicAnalysis[] {
    return Array.from(this.strategyDatabase.values());
  }

  /**
   * Get available tactics
   */
  getAvailableTactics(): TacticalRecommendation[] {
    return Array.from(this.tacticLibrary.values());
  }

  /**
   * Get game theory model for specific type
   */
  getGameTheoryModel(gameType: GameType): GameTheoryModel | undefined {
    return this.gameTheoryModels.get(gameType);
  }

  /**
   * Dispose of engine resources
   */
  dispose(): void {
    this.strategyDatabase.clear();
    this.tacticLibrary.clear();
    this.gameTheoryModels.clear();
    this.removeAllListeners();
    this.emit('engineDisposed');
  }
}