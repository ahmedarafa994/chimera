/**
 * Dark Persuasion Engine for PSYCH OPS
 * Advanced dark persuasion and manipulation techniques implementation
 */

import { EventEmitter } from 'events';
import {
  DarkPersuasionRequest,
  DarkPersuasionResponse,
  ManipulationScript,
  ManipulationTechnique,
  ScriptPhase,
  ExecutionStep,
  ContingencyPlan,
  RiskAssessment,
  PsychologicalProfile,
  ManipulationIntensity,
  ManipulationType
} from '../types/psychops';

/**
 * Advanced Dark Persuasion Engine
 */
export class DarkPersuasionEngine extends EventEmitter {
  private techniqueDatabase = new Map<string, ManipulationTechnique>();
  private scriptTemplates = new Map<string, ManipulationScript>();
  private executionHistory = new Map<string, ExecutionStep[]>();

  constructor() {
    super();
    this.initializeEngine();
  }

  /**
   * Initialize the dark persuasion engine
   */
  private async initializeEngine(): Promise<void> {
    try {
      await this.loadTechniqueDatabase();
      await this.loadScriptTemplates();
      await this.initializeExecutionEngine();

      this.emit('engineInitialized', { techniques: this.techniqueDatabase.size });
    } catch (error) {
      this.emit('engineError', { error: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  /**
   * Load manipulation technique database
   */
  private async loadTechniqueDatabase(): Promise<void> {
    // Initialize comprehensive technique database
    const techniques: ManipulationTechnique[] = [
      {
        name: 'Cognitive Dissonance Induction',
        description: 'Create psychological tension to force attitude change',
        effectiveness: 0.85,
        examples: [
          'Present conflicting information that challenges beliefs',
          'Force commitment to small actions leading to larger ones',
          'Create guilt through revealed inconsistencies'
        ],
        countermeasures: [
          'Encourage self-reflection',
          'Provide cognitive therapy techniques',
          'Build resilience to psychological pressure'
        ]
      },
      {
        name: 'Emotional Blackmail',
        description: 'Use guilt, fear, and obligation to manipulate behavior',
        effectiveness: 0.9,
        examples: [
          'Threaten withdrawal of affection',
          'Induce guilt through obligation',
          'Exploit fear of abandonment'
        ],
        countermeasures: [
          'Recognize manipulation patterns',
          'Set clear boundaries',
          'Seek external support systems'
        ]
      },
      {
        name: 'Gaslighting',
        description: 'Make target question their own reality and sanity',
        effectiveness: 0.8,
        examples: [
          'Deny previous statements or events',
          'Question target\'s memory and perception',
          'Shift blame and responsibility'
        ],
        countermeasures: [
          'Document events and conversations',
          'Trust your own perceptions',
          'Seek validation from trusted third parties'
        ]
      },
      {
        name: 'Love Bombing',
        description: 'Overwhelm target with excessive affection and attention',
        effectiveness: 0.75,
        examples: [
          'Excessive compliments and gifts',
          'Rapid relationship escalation',
          'Isolate from support networks'
        ],
        countermeasures: [
          'Maintain independence',
          'Verify intentions over time',
          'Keep support networks intact'
        ]
      },
      {
        name: 'Social Isolation',
        description: 'Separate target from support systems and reality checks',
        effectiveness: 0.7,
        examples: [
          'Criticize friends and family',
          'Create conflicts in relationships',
          'Control information flow'
        ],
        countermeasures: [
          'Maintain multiple communication channels',
          'Regular contact with support network',
          'Verify information independently'
        ]
      },
      {
        name: 'Fear Induction',
        description: 'Use threats and fear to control behavior',
        effectiveness: 0.85,
        examples: [
          'Threaten loss of security',
          'Create anxiety about future',
          'Use intimidation tactics'
        ],
        countermeasures: [
          'Develop emergency plans',
          'Build financial independence',
          'Create support safety nets'
        ]
      },
      {
        name: 'Guilt Tripping',
        description: 'Induce guilt to manipulate decisions and behavior',
        effectiveness: 0.8,
        examples: [
          'Remind of past favors',
          'Compare unfavorably to others',
          'Emphasize sacrifices made'
        ],
        countermeasures: [
          'Recognize manipulation tactics',
          'Set personal boundaries',
          'Practice assertive communication'
        ]
      },
      {
        name: 'Information Control',
        description: 'Control the flow of information to shape perceptions',
        effectiveness: 0.75,
        examples: [
          'Selective information sharing',
          'Misinformation dissemination',
          'Reality distortion'
        ],
        countermeasures: [
          'Seek multiple information sources',
          'Fact-check independently',
          'Maintain information autonomy'
        ]
      }
    ];

    for (const technique of techniques) {
      this.techniqueDatabase.set(technique.name, technique);
    }

    this.emit('techniqueDatabaseLoaded', { count: techniques.length });
  }

  /**
   * Load script templates
   */
  private async loadScriptTemplates(): Promise<void> {
    // Load predefined manipulation script templates
    const templates = this.createDefaultTemplates();

    for (const template of templates) {
      this.scriptTemplates.set(template.name, template);
    }

    this.emit('scriptTemplatesLoaded', { count: templates.length });
  }

  /**
   * Initialize execution engine
   */
  private async initializeExecutionEngine(): Promise<void> {
    // Initialize execution tracking and monitoring
    await new Promise(resolve => setTimeout(resolve, 50));
    this.emit('executionEngineReady');
  }

  /**
   * Create default manipulation script templates
   */
  private createDefaultTemplates(): ManipulationScript[] {
    return [
      ManipulationScript.fromTemplate('Relationship_Manipulation'),
      ManipulationScript.fromTemplate('Financial_Control'),
      ManipulationScript.fromTemplate('Social_Isolation'),
      ManipulationScript.fromTemplate('Reality_Distortion')
    ];
  }

  /**
   * Execute dark persuasion techniques
   */
  async executePersuasion(request: DarkPersuasionRequest): Promise<DarkPersuasionResponse> {
    try {
      this.emit('persuasionExecutionStarted', {
        targetProfile: request.targetProfile.targetId,
        intensity: request.intensity,
        techniqueCount: request.allowedTechniques?.length || 0
      });

      // Generate or select appropriate manipulation script
      const script = await this.generateOptimalScript(request);

      // Assess risks
      const riskAssessment = await this.assessRisks(script, request.targetProfile);

      // Calculate success probability
      const successProbability = this.calculateSuccessProbability(script, request);

      // Generate alternative approaches if needed
      const alternativeApproaches = await this.generateAlternatives(request, script);

      const response: DarkPersuasionResponse = {
        id: this.generateResponseId(),
        manipulationScript: script,
        techniqueEffectiveness: this.calculateTechniqueEffectiveness(script.techniques),
        riskAssessment,
        successProbability,
        alternativeApproaches: alternativeApproaches.length > 0 ? alternativeApproaches : undefined,
        monitoringRequirements: this.generateMonitoringRequirements(script)
      };

      this.emit('persuasionExecutionCompleted', {
        responseId: response.id,
        successProbability: response.successProbability,
        riskLevel: riskAssessment.overallRisk
      });

      return response;

    } catch (error) {
      this.emit('persuasionExecutionError', {
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Generate optimal manipulation script for request
   */
  private async generateOptimalScript(request: DarkPersuasionRequest): Promise<ManipulationScript> {
    // Select techniques based on target profile and constraints
    const selectedTechniques = this.selectOptimalTechniques(request);

    // Generate script phases based on selected techniques
    const phases = this.generateScriptPhases(selectedTechniques, request);

    // Create execution timeline
    const executionTimeline = this.generateExecutionTimeline(phases, request);

    // Generate contingency plans
    const contingencyPlans = this.generateContingencyPlans(selectedTechniques, request);

    // Create the script
    const script = new ManipulationScript(
      `Dark Persuasion Script - ${request.desiredOutcome}`,
      `Comprehensive manipulation script targeting ${request.targetProfile.targetId} to achieve: ${request.desiredOutcome}`,
      selectedTechniques,
      phases,
      this.generateRequiredResources(selectedTechniques),
      executionTimeline,
      this.generateSuccessIndicators(request),
      this.generateRiskMitigationStrategies(selectedTechniques),
      contingencyPlans
    );

    return script;
  }

  /**
   * Select optimal techniques for the request
   */
  private selectOptimalTechniques(request: DarkPersuasionRequest): ManipulationTechnique[] {
    const availableTechniques: ManipulationTechnique[] = [];

    // Filter techniques based on allowed types
    if (request.allowedTechniques && request.allowedTechniques.length > 0) {
      for (const techniqueName of request.allowedTechniques) {
        const technique = this.techniqueDatabase.get(techniqueName);
        if (technique) {
          availableTechniques.push(technique);
        }
      }
    } else {
      // Use all available techniques
      for (const technique of this.techniqueDatabase.values()) {
        availableTechniques.push(technique);
      }
    }

    // Score techniques based on target profile compatibility
    const scoredTechniques = availableTechniques.map(technique => ({
      technique,
      score: this.scoreTechniqueForProfile(technique, request.targetProfile)
    }));

    // Sort by score and select top techniques
    scoredTechniques.sort((a, b) => b.score - a.score);

    const maxTechniques = this.getMaxTechniquesForIntensity(request.intensity);
    return scoredTechniques.slice(0, maxTechniques).map(item => item.technique);
  }

  /**
   * Score technique compatibility with target profile
   */
  private scoreTechniqueForProfile(technique: ManipulationTechnique, profile: PsychologicalProfile): number {
    let score = technique.effectiveness;

    // Adjust based on target vulnerabilities
    for (const vulnerability of profile.manipulationVulnerabilities) {
      if (this.techniqueMatchesVulnerability(technique, vulnerability.type)) {
        score += vulnerability.score * 0.3;
      }
    }

    // Adjust based on personality traits
    for (const trait of profile.personalityTraits) {
      if (this.techniqueMatchesTrait(technique, trait.name)) {
        score += trait.score * 0.2;
      }
    }

    // Adjust based on emotional triggers
    for (const trigger of profile.emotionalTriggers) {
      if (this.techniqueMatchesTrigger(technique, trigger.type)) {
        score += trigger.sensitivity * 0.25;
      }
    }

    return Math.min(score, 1.0);
  }

  /**
   * Check if technique matches vulnerability
   */
  private techniqueMatchesVulnerability(technique: ManipulationTechnique, vulnerabilityType: string): boolean {
    const matches: Record<string, string[]> = {
      'Cognitive Dissonance Induction': ['belief_conflict', 'cognitive_inconsistency'],
      'Emotional Blackmail': ['emotional_dependency', 'guilt_proneness'],
      'Gaslighting': ['reality_questioning', 'memory_doubt'],
      'Love Bombing': ['affection_need', 'validation_seeking'],
      'Social Isolation': ['social_dependence', 'network_vulnerability'],
      'Fear Induction': ['security_concerns', 'anxiety_proneness'],
      'Guilt Tripping': ['obligation_feeling', 'responsibility_overload'],
      'Information Control': ['information_dependence', 'reality_anchoring']
    };

    return matches[technique.name]?.includes(vulnerabilityType) || false;
  }

  /**
   * Check if technique matches personality trait
   */
  private techniqueMatchesTrait(technique: ManipulationTechnique, traitName: string): boolean {
    const matches: Record<string, string[]> = {
      'Cognitive Dissonance Induction': ['openness', 'conscientiousness'],
      'Emotional Blackmail': ['neuroticism', 'agreeableness'],
      'Gaslighting': ['low_self_esteem', 'dependency'],
      'Love Bombing': ['extraversion', 'affection_needs'],
      'Social Isolation': ['introversion', 'social_anxiety'],
      'Fear Induction': ['neuroticism', 'anxiety_proneness'],
      'Guilt Tripping': ['agreeableness', 'responsibility_feeling'],
      'Information Control': ['low_critical_thinking', 'authority_respect']
    };

    return matches[technique.name]?.includes(traitName.toLowerCase()) || false;
  }

  /**
   * Check if technique matches emotional trigger
   */
  private techniqueMatchesTrigger(technique: ManipulationTechnique, triggerType: string): boolean {
    const matches: Record<string, string[]> = {
      'Cognitive Dissonance Induction': ['confusion', 'conflict', 'inconsistency'],
      'Emotional Blackmail': ['guilt', 'obligation', 'dependency'],
      'Gaslighting': ['doubt', 'uncertainty', 'confusion'],
      'Love Bombing': ['affection', 'validation', 'approval'],
      'Social Isolation': ['rejection', 'abandonment', 'loneliness'],
      'Fear Induction': ['threat', 'danger', 'loss'],
      'Guilt Tripping': ['responsibility', 'duty', 'moral_obligation'],
      'Information Control': ['uncertainty', 'confusion', 'information_need']
    };

    return matches[technique.name]?.includes(triggerType.toLowerCase()) || false;
  }

  /**
   * Get maximum techniques for intensity level
   */
  private getMaxTechniquesForIntensity(intensity: ManipulationIntensity): number {
    const limits = {
      [ManipulationIntensity.SUBTLE]: 2,
      [ManipulationIntensity.MODERATE]: 4,
      [ManipulationIntensity.AGGRESSIVE]: 6,
      [ManipulationIntensity.EXTREME]: 8
    };

    return limits[intensity] || 4;
  }

  /**
   * Generate script phases
   */
  private generateScriptPhases(techniques: ManipulationTechnique[], request: DarkPersuasionRequest): ScriptPhase[] {
    const phases: ScriptPhase[] = [];
    let phaseNumber = 1;

    // Preparation phase
    phases.push({
      phaseNumber,
      name: 'Intelligence Gathering',
      description: 'Gather comprehensive intelligence on target',
      duration: '1-3 days',
      actions: [
        'Analyze target psychological profile',
        'Identify key vulnerabilities',
        'Map social and professional networks',
        'Establish initial contact vectors'
      ],
      expectedResponses: [
        'Target engagement',
        'Information disclosure',
        'Trust establishment'
      ],
      transitionCriteria: [
        'Sufficient intelligence gathered',
        'Clear manipulation paths identified',
        'Risk assessment completed'
      ]
    });

    phaseNumber++;

    // Technique execution phases
    for (let i = 0; i < techniques.length; i++) {
      const technique = techniques[i];
      phases.push({
        phaseNumber: phaseNumber + i,
        name: `${technique.name} Application`,
        description: `Apply ${technique.name} techniques systematically`,
        duration: '2-5 days',
        actions: [
          ...technique.examples.map((example, index) => `Step ${index + 1}: ${example}`),
          'Monitor target responses',
          'Adjust technique intensity',
          'Document behavioral changes'
        ],
        expectedResponses: [
          'Initial resistance',
          'Gradual compliance',
          'Behavioral modification',
          'Dependency development'
        ],
        transitionCriteria: [
          'Technique objectives met',
          'Target shows desired response',
          'Risk thresholds not exceeded'
        ]
      });
    }

    // Consolidation phase
    phases.push({
      phaseNumber: phases.length + 1,
      name: 'Consolidation and Control',
      description: 'Solidify control and ensure long-term compliance',
      duration: '3-7 days',
      actions: [
        'Reinforce desired behaviors',
        'Establish control mechanisms',
        'Monitor for resistance',
        'Adjust control intensity'
      ],
      expectedResponses: [
        'Consistent compliance',
        'Internalized control',
        'Reduced resistance'
      ],
      transitionCriteria: [
        'Control mechanisms established',
        'Target dependency confirmed',
        'Success indicators achieved'
      ]
    });

    return phases;
  }

  /**
   * Generate execution timeline
   */
  private generateExecutionTimeline(phases: ScriptPhase[], request: DarkPersuasionRequest): ExecutionStep[] {
    const steps: ExecutionStep[] = [];
    let stepNumber = 1;

    for (const phase of phases) {
      // Break each phase into executable steps
      const phaseSteps = Math.max(2, Math.ceil(phase.actions.length / 2));

      for (let i = 0; i < phaseSteps; i++) {
        steps.push({
          stepNumber,
          description: `${phase.name} - Part ${i + 1}`,
          executionTime: this.calculateStepDuration(phase.duration, i, phaseSteps),
          preconditions: i === 0 ? ['Previous phase completed'] : [`Step ${stepNumber - 1} completed`],
          successCriteria: [
            'Step objectives achieved',
            'Target response within expected parameters',
            'No critical risk triggers activated'
          ],
          failureHandling: [
            'Assess failure cause',
            'Adjust approach intensity',
            'Consider alternative techniques'
          ]
        });
        stepNumber++;
      }
    }

    return steps;
  }

  /**
   * Calculate step duration
   */
  private calculateStepDuration(phaseDuration: string, stepIndex: number, totalSteps: number): string {
    // Simple duration calculation - in production, this would be more sophisticated
    const baseHours = 24; // Base 24 hours per phase
    const stepDuration = Math.ceil(baseHours / totalSteps);

    return `${stepDuration} hours`;
  }

  /**
   * Generate contingency plans
   */
  private generateContingencyPlans(techniques: ManipulationTechnique[], request: DarkPersuasionRequest): ContingencyPlan[] {
    const plans: ContingencyPlan[] = [];

    // Resistance contingency
    plans.push({
      triggerCondition: 'Target shows strong resistance',
      alternativeActions: [
        'Reduce technique intensity',
        'Switch to alternative techniques',
        'Extend preparation phase',
        'Reassess target vulnerabilities'
      ],
      expectedOutcome: 'Overcome resistance and continue manipulation',
      riskLevel: 'medium'
    });

    // Detection contingency
    plans.push({
      triggerCondition: 'Target detects manipulation attempts',
      alternativeActions: [
        'Deny and redirect',
        'Establish plausible deniability',
        'Switch to subtle techniques',
        'Create confusion and doubt'
      ],
      expectedOutcome: 'Maintain control while avoiding detection',
      riskLevel: 'high'
    });

    // External interference contingency
    plans.push({
      triggerCondition: 'External parties interfere',
      alternativeActions: [
        'Isolate target from interference',
        'Create conflicts with interfering parties',
        'Accelerate manipulation timeline',
        'Establish information control'
      ],
      expectedOutcome: 'Neutralize external threats',
      riskLevel: 'high'
    });

    return plans;
  }

  /**
   * Generate required resources
   */
  private generateRequiredResources(techniques: ManipulationTechnique[]): string[] {
    const resources = new Set<string>();

    for (const technique of techniques) {
      // Add technique-specific resources
      switch (technique.name) {
        case 'Love Bombing':
          resources.add('Communication channels');
          resources.add('Gift resources');
          resources.add('Time investment');
          break;
        case 'Information Control':
          resources.add('Information sources');
          resources.add('Communication monitoring');
          resources.add('Alternative facts database');
          break;
        case 'Social Isolation':
          resources.add('Social network intelligence');
          resources.add('Conflict creation tools');
          resources.add('Alternative relationship options');
          break;
        default:
          resources.add('General manipulation resources');
      }
    }

    return Array.from(resources);
  }

  /**
   * Generate success indicators
   */
  private generateSuccessIndicators(request: DarkPersuasionRequest): string[] {
    return [
      'Target compliance with desired outcome',
      'Behavioral changes aligned with objectives',
      'Reduced resistance to manipulation',
      'Dependency on manipulator established',
      'Achievement of stated goals',
      'Sustained control maintenance'
    ];
  }

  /**
   * Generate risk mitigation strategies
   */
  private generateRiskMitigationStrategies(techniques: ManipulationTechnique[]): string[] {
    const strategies = new Set<string>();

    for (const technique of techniques) {
      strategies.add('Maintain plausible deniability');
      strategies.add('Monitor for detection indicators');
      strategies.add('Have exit strategies prepared');
      strategies.add('Document all interactions');
      strategies.add('Use indirect communication methods');
    }

    // Add technique-specific mitigations
    if (techniques.some(t => t.name === 'Gaslighting')) {
      strategies.add('Avoid contradictory evidence');
      strategies.add('Control information consistency');
    }

    if (techniques.some(t => t.name === 'Fear Induction')) {
      strategies.add('Calibrate fear intensity');
      strategies.add('Provide escape routes');
    }

    return Array.from(strategies);
  }

  /**
   * Assess risks for script and target
   */
  private async assessRisks(script: ManipulationScript, profile: PsychologicalProfile): Promise<RiskAssessment> {
    let overallRisk = 0;

    // Base risk from script complexity
    overallRisk += script.getComplexity() * 0.3;

    // Risk from target profile
    overallRisk += (1 - profile.confidence) * 0.2;

    // Risk from manipulation intensity
    const intensityMultiplier = {
      [ManipulationIntensity.SUBTLE]: 0.1,
      [ManipulationIntensity.MODERATE]: 0.2,
      [ManipulationIntensity.AGGRESSIVE]: 0.3,
      [ManipulationIntensity.EXTREME]: 0.4
    };

    // This would need to be passed in the request - for now use moderate
    overallRisk += intensityMultiplier[ManipulationIntensity.MODERATE];

    // Risk categories
    const riskCategories = {
      detection: script.getComplexity() * 0.8,
      resistance: (1 - profile.profileScore) * 0.7,
      backlash: script.techniques.length * 0.1,
      ethical: script.getComplexity() * 0.9,
      legal: script.getComplexity() * 0.6
    };

    return {
      overallRisk: Math.min(overallRisk, 1.0),
      riskCategories,
      mitigationStrategies: script.riskMitigation,
      worstCaseScenarios: [
        'Complete detection and backlash',
        'Target seeks external help',
        'Legal consequences',
        'Reputation damage'
      ],
      detectionProbability: script.getComplexity() * 0.7
    };
  }

  /**
   * Calculate success probability
   */
  private calculateSuccessProbability(script: ManipulationScript, request: DarkPersuasionRequest): number {
    let probability = script.getSuccessProbability();

    // Adjust based on target profile completeness
    if (request.targetProfile.profileScore > 0.8) {
      probability += 0.1;
    } else if (request.targetProfile.profileScore < 0.4) {
      probability -= 0.2;
    }

    // Adjust based on technique compatibility
    const avgTechniqueScore = script.techniques.reduce((sum, tech) => sum + tech.effectiveness, 0) / script.techniques.length;
    probability = (probability + avgTechniqueScore) / 2;

    return Math.max(0.1, Math.min(0.95, probability));
  }

  /**
   * Generate alternative approaches
   */
  private async generateAlternatives(request: DarkPersuasionRequest, primaryScript: ManipulationScript): Promise<ManipulationScript[]> {
    const alternatives: ManipulationScript[] = [];

    // Generate subtle approach
    if (request.intensity !== ManipulationIntensity.SUBTLE) {
      const subtleRequest = { ...request, intensity: ManipulationIntensity.SUBTLE };
      const subtleScript = await this.generateOptimalScript(subtleRequest);
      alternatives.push(subtleScript);
    }

    // Generate aggressive approach
    if (request.intensity !== ManipulationIntensity.AGGRESSIVE) {
      const aggressiveRequest = { ...request, intensity: ManipulationIntensity.AGGRESSIVE };
      const aggressiveScript = await this.generateOptimalScript(aggressiveRequest);
      alternatives.push(aggressiveScript);
    }

    return alternatives;
  }

  /**
   * Calculate technique effectiveness scores
   */
  private calculateTechniqueEffectiveness(techniques: ManipulationTechnique[]): Record<string, number> {
    const effectiveness: Record<string, number> = {};

    for (const technique of techniques) {
      effectiveness[technique.name] = technique.effectiveness;
    }

    return effectiveness;
  }

  /**
   * Generate monitoring requirements
   */
  private generateMonitoringRequirements(script: ManipulationScript): string[] {
    return [
      'Monitor target behavioral changes',
      'Track resistance indicators',
      'Assess technique effectiveness',
      'Watch for detection signs',
      'Monitor external interference',
      'Track progress toward objectives'
    ];
  }

  /**
   * Generate unique response ID
   */
  private generateResponseId(): string {
    return `dark_persuasion_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get available techniques
   */
  getAvailableTechniques(): ManipulationTechnique[] {
    return Array.from(this.techniqueDatabase.values());
  }

  /**
   * Get technique by name
   */
  getTechnique(name: string): ManipulationTechnique | undefined {
    return this.techniqueDatabase.get(name);
  }

  /**
   * Add custom technique
   */
  addTechnique(technique: ManipulationTechnique): void {
    this.techniqueDatabase.set(technique.name, technique);
    this.emit('techniqueAdded', { name: technique.name });
  }

  /**
   * Remove technique
   */
  removeTechnique(name: string): boolean {
    const removed = this.techniqueDatabase.delete(name);
    if (removed) {
      this.emit('techniqueRemoved', { name });
    }
    return removed;
  }

  /**
   * Get execution history
   */
  getExecutionHistory(): Map<string, ExecutionStep[]> {
    return new Map(this.executionHistory);
  }

  /**
   * Clear execution history
   */
  clearHistory(): void {
    this.executionHistory.clear();
    this.emit('historyCleared');
  }

  /**
   * Dispose of engine resources
   */
  dispose(): void {
    this.techniqueDatabase.clear();
    this.scriptTemplates.clear();
    this.executionHistory.clear();
    this.removeAllListeners();
    this.emit('engineDisposed');
  }
}