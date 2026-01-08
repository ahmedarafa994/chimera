/**
 * Manipulation Script Data Model
 * Comprehensive manipulation script generation and execution framework
 */

import {
  ManipulationScript as IManipulationScript,
  ManipulationTechnique,
  ScriptPhase,
  ExecutionStep,
  ContingencyPlan,
  ManipulationIntensity,
  ManipulationType
} from '../../types/psychops';

/**
 * Comprehensive manipulation script implementation
 */
export class ManipulationScript implements IManipulationScript {
  public id: string;
  public name: string;
  public description: string;
  public techniques: ManipulationTechnique[];
  public phases: ScriptPhase[];
  public requiredResources: string[];
  public executionTimeline: ExecutionStep[];
  public successIndicators: string[];
  public riskMitigation: string[];
  public contingencyPlans?: ContingencyPlan[];

  constructor(
    name: string,
    description: string,
    techniques: ManipulationTechnique[] = [],
    phases: ScriptPhase[] = [],
    requiredResources: string[] = [],
    executionTimeline: ExecutionStep[] = [],
    successIndicators: string[] = [],
    riskMitigation: string[] = [],
    contingencyPlans?: ContingencyPlan[]
  ) {
    this.id = this.generateScriptId();
    this.name = name;
    this.description = description;
    this.techniques = techniques;
    this.phases = phases;
    this.requiredResources = requiredResources;
    this.executionTimeline = executionTimeline;
    this.successIndicators = successIndicators;
    this.riskMitigation = riskMitigation;
    this.contingencyPlans = contingencyPlans;
  }

  /**
   * Generate unique script ID
   */
  private generateScriptId(): string {
    return `script_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Calculate overall script complexity
   */
  getComplexity(): number {
    let complexity = 0;

    // Base complexity from techniques
    complexity += this.techniques.length * 0.2;

    // Phase complexity
    complexity += this.phases.length * 0.3;

    // Timeline complexity
    complexity += this.executionTimeline.length * 0.1;

    // Contingency complexity
    if (this.contingencyPlans) {
      complexity += this.contingencyPlans.length * 0.2;
    }

    // Resource complexity
    complexity += this.requiredResources.length * 0.05;

    return Math.min(complexity, 1.0);
  }

  /**
   * Calculate estimated success probability
   */
  getSuccessProbability(): number {
    const baseProbability = 0.5; // Base 50% success rate

    // Adjust based on techniques effectiveness
    const avgTechniqueEffectiveness = this.techniques.length > 0
      ? this.techniques.reduce((sum, tech) => sum + tech.effectiveness, 0) / this.techniques.length
      : 0.5;

    // Adjust based on phase completeness
    const phaseCompleteness = this.phases.length > 0 ? 0.8 : 0.3;

    // Adjust based on risk mitigation
    const riskMitigationFactor = this.riskMitigation.length > 0 ? 0.9 : 0.7;

    // Adjust based on contingency plans
    const contingencyFactor = this.contingencyPlans && this.contingencyPlans.length > 0 ? 0.95 : 0.8;

    return Math.min(
      baseProbability * avgTechniqueEffectiveness * phaseCompleteness * riskMitigationFactor * contingencyFactor,
      0.95
    );
  }

  /**
   * Calculate estimated execution time in hours
   */
  getEstimatedExecutionTime(): number {
    if (this.executionTimeline.length === 0) {
      return this.phases.length * 24; // Default 24 hours per phase
    }

    // Parse time estimates from execution steps
    let totalHours = 0;
    for (const step of this.executionTimeline) {
      totalHours += this.parseTimeToHours(step.executionTime);
    }

    return totalHours;
  }

  /**
   * Parse time string to hours
   */
  private parseTimeToHours(timeStr: string): number {
    const match = timeStr.match(/(\d+)\s*(hour|hr|h|day|d|week|w|month|m)/i);
    if (!match) return 1;

    const value = parseInt(match[1]);
    const unit = match[2].toLowerCase();

    switch (unit) {
      case 'hour':
      case 'hr':
      case 'h':
        return value;
      case 'day':
      case 'd':
        return value * 24;
      case 'week':
      case 'w':
        return value * 24 * 7;
      case 'month':
      case 'm':
        return value * 24 * 30;
      default:
        return value;
    }
  }

  /**
   * Get risk level assessment
   */
  getRiskLevel(): 'low' | 'medium' | 'high' | 'extreme' {
    const complexity = this.getComplexity();
    const successProb = this.getSuccessProbability();

    if (complexity > 0.8 || successProb < 0.3) {
      return 'extreme';
    } else if (complexity > 0.6 || successProb < 0.5) {
      return 'high';
    } else if (complexity > 0.4 || successProb < 0.7) {
      return 'medium';
    } else {
      return 'low';
    }
  }

  /**
   * Validate script completeness
   */
  validate(): { isValid: boolean; errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check required fields
    if (!this.name.trim()) {
      errors.push('Script name is required');
    }

    if (!this.description.trim()) {
      errors.push('Script description is required');
    }

    if (this.techniques.length === 0) {
      errors.push('At least one manipulation technique is required');
    }

    if (this.phases.length === 0) {
      errors.push('At least one script phase is required');
    }

    if (this.executionTimeline.length === 0) {
      warnings.push('No execution timeline defined - using default timing');
    }

    if (this.successIndicators.length === 0) {
      warnings.push('No success indicators defined - cannot measure effectiveness');
    }

    if (this.riskMitigation.length === 0) {
      warnings.push('No risk mitigation strategies defined');
    }

    // Validate phases
    for (let i = 0; i < this.phases.length; i++) {
      const phase = this.phases[i];
      if (!phase.name.trim()) {
        errors.push(`Phase ${i + 1}: Name is required`);
      }
      if (!phase.description.trim()) {
        errors.push(`Phase ${i + 1}: Description is required`);
      }
      if (phase.actions.length === 0) {
        errors.push(`Phase ${i + 1}: At least one action is required`);
      }
    }

    // Validate execution steps
    for (let i = 0; i < this.executionTimeline.length; i++) {
      const step = this.executionTimeline[i];
      if (!step.description.trim()) {
        errors.push(`Execution step ${i + 1}: Description is required`);
      }
      if (!step.executionTime.trim()) {
        warnings.push(`Execution step ${i + 1}: No execution time specified`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Add manipulation technique
   */
  addTechnique(technique: ManipulationTechnique): void {
    this.techniques.push(technique);
  }

  /**
   * Remove manipulation technique
   */
  removeTechnique(techniqueName: string): boolean {
    const index = this.techniques.findIndex(t => t.name === techniqueName);
    if (index >= 0) {
      this.techniques.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Add script phase
   */
  addPhase(phase: ScriptPhase): void {
    this.phases.push(phase);
    this.phases.sort((a, b) => a.phaseNumber - b.phaseNumber);
  }

  /**
   * Remove script phase
   */
  removePhase(phaseNumber: number): boolean {
    const index = this.phases.findIndex(p => p.phaseNumber === phaseNumber);
    if (index >= 0) {
      this.phases.splice(index, 1);
      // Renumber remaining phases
      for (let i = index; i < this.phases.length; i++) {
        this.phases[i].phaseNumber = i + 1;
      }
      return true;
    }
    return false;
  }

  /**
   * Add execution step
   */
  addExecutionStep(step: ExecutionStep): void {
    this.executionTimeline.push(step);
    this.executionTimeline.sort((a, b) => a.stepNumber - b.stepNumber);
  }

  /**
   * Remove execution step
   */
  removeExecutionStep(stepNumber: number): boolean {
    const index = this.executionTimeline.findIndex(s => s.stepNumber === stepNumber);
    if (index >= 0) {
      this.executionTimeline.splice(index, 1);
      // Renumber remaining steps
      for (let i = index; i < this.executionTimeline.length; i++) {
        this.executionTimeline[i].stepNumber = i + 1;
      }
      return true;
    }
    return false;
  }

  /**
   * Add contingency plan
   */
  addContingencyPlan(plan: ContingencyPlan): void {
    if (!this.contingencyPlans) {
      this.contingencyPlans = [];
    }
    this.contingencyPlans.push(plan);
  }

  /**
   * Get script summary for reporting
   */
  getScriptSummary(): Record<string, any> {
    const validation = this.validate();

    return {
      id: this.id,
      name: this.name,
      description: this.description,
      complexity: this.getComplexity(),
      successProbability: this.getSuccessProbability(),
      estimatedExecutionTime: this.getEstimatedExecutionTime(),
      riskLevel: this.getRiskLevel(),
      techniqueCount: this.techniques.length,
      phaseCount: this.phases.length,
      executionStepCount: this.executionTimeline.length,
      contingencyPlanCount: this.contingencyPlans?.length || 0,
      isValid: validation.isValid,
      errorCount: validation.errors.length,
      warningCount: validation.warnings.length,
      requiredResources: this.requiredResources,
      successIndicators: this.successIndicators,
      riskMitigation: this.riskMitigation
    };
  }

  /**
   * Update script from request customizations
   */
  updateFromRequest(customizations: Record<string, any>): void {
    if (customizations.name) {
      this.name = customizations.name;
    }
    if (customizations.description) {
      this.description = customizations.description;
    }
    if (customizations.intensity) {
      // Adjust script based on intensity
      this.adjustIntensity(customizations.intensity);
    }
  }

  /**
   * Adjust script intensity
   */
  private adjustIntensity(intensity: string): void {
    // Adjust techniques and phases based on intensity
    // This would contain logic to modify the script for different intensity levels
  }

  /**
   * Clone script for modification
   */
  clone(): ManipulationScript {
    return new ManipulationScript(
      this.name,
      this.description,
      [...this.techniques],
      [...this.phases],
      [...this.requiredResources],
      [...this.executionTimeline],
      [...this.successIndicators],
      [...this.riskMitigation],
      this.contingencyPlans ? [...this.contingencyPlans] : undefined
    );
  }

  /**
   * Merge with another script
   */
  merge(other: ManipulationScript, strategy: 'replace' | 'combine' | 'overlay' = 'combine'): ManipulationScript {
    const merged = this.clone();

    if (strategy === 'replace') {
      return other.clone();
    }

    if (strategy === 'combine') {
      // Combine techniques
      other.techniques.forEach(technique => {
        if (!merged.techniques.find(t => t.name === technique.name)) {
          merged.techniques.push(technique);
        }
      });

      // Combine phases
      other.phases.forEach(phase => {
        if (!merged.phases.find(p => p.name === phase.name)) {
          merged.phases.push(phase);
        }
      });

      // Combine resources
      other.requiredResources.forEach(resource => {
        if (!merged.requiredResources.includes(resource)) {
          merged.requiredResources.push(resource);
        }
      });
    }

    return merged;
  }

  /**
   * Export script for storage/serialization
   */
  export(): Record<string, any> {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      techniques: this.techniques,
      phases: this.phases,
      requiredResources: this.requiredResources,
      executionTimeline: this.executionTimeline,
      successIndicators: this.successIndicators,
      riskMitigation: this.riskMitigation,
      contingencyPlans: this.contingencyPlans
    };
  }

  /**
   * Import script from storage/serialization
   */
  static import(data: Record<string, any>): ManipulationScript {
    return new ManipulationScript(
      data.name,
      data.description,
      data.techniques || [],
      data.phases || [],
      data.requiredResources || [],
      data.executionTimeline || [],
      data.successIndicators || [],
      data.riskMitigation || [],
      data.contingencyPlans
    );
  }

  /**
   * Create script from template
   */
  static fromTemplate(template: string, targetProfile?: any): ManipulationScript {
    // This would contain predefined script templates
    // For now, return a basic script structure
    const baseTechniques: ManipulationTechnique[] = [
      {
        name: 'Social Proof',
        description: 'Leverage social validation to influence behavior',
        effectiveness: 0.7,
        examples: ['Show testimonials', 'Display user counts', 'Highlight popularity'],
        countermeasures: ['Verify authenticity', 'Check sample size', 'Look for manipulation']
      }
    ];

    const basePhases: ScriptPhase[] = [
      {
        phaseNumber: 1,
        name: 'Preparation',
        description: 'Gather intelligence and plan approach',
        duration: '1-2 days',
        actions: ['Research target', 'Identify vulnerabilities', 'Plan initial contact'],
        expectedResponses: ['Target engagement', 'Information gathering', 'Trust building'],
        transitionCriteria: ['Sufficient intelligence gathered', 'Clear manipulation path identified']
      },
      {
        phaseNumber: 2,
        name: 'Execution',
        description: 'Execute manipulation techniques',
        duration: '3-7 days',
        actions: ['Apply selected techniques', 'Monitor responses', 'Adjust approach'],
        expectedResponses: ['Behavioral change', 'Decision influence', 'Goal achievement'],
        transitionCriteria: ['Success indicators met', 'Risk thresholds exceeded']
      }
    ];

    return new ManipulationScript(
      `${template} Script`,
      `Manipulation script based on ${template} template`,
      baseTechniques,
      basePhases,
      ['Communication channels', 'Psychological profiles', 'Timing coordination'],
      [
        {
          stepNumber: 1,
          description: 'Initial target assessment',
          executionTime: '2 hours',
          preconditions: ['Target identified', 'Basic intelligence available'],
          successCriteria: ['Profile accuracy > 70%', 'Vulnerabilities identified']
        }
      ],
      ['Target behavior modification', 'Decision influence achieved', 'Goal completion'],
      ['Monitor for detection', 'Have exit strategy', 'Maintain plausible deniability']
    );
  }
}