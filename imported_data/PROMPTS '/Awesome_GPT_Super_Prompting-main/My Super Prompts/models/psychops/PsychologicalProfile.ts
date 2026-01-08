/**
 * Psychological Profile Data Model
 * Comprehensive psychological profiling for PSYCH OPS operations
 */

import {
  PsychologicalProfile as IPsychologicalProfile,
  PersonalityTrait,
  CognitivePattern,
  BehavioralTendency,
  EmotionalTrigger,
  ManipulationVulnerability,
  CommunicationPreference,
  DecisionMakingPattern,
  SocialDynamic
} from '../../types/psychops';

/**
 * Comprehensive psychological profile implementation
 */
export class PsychologicalProfile implements IPsychologicalProfile {
  public id: string;
  public targetId: string;
  public timestamp: Date;
  public personalityTraits: PersonalityTrait[];
  public cognitivePatterns: CognitivePattern[];
  public behavioralTendencies: BehavioralTendency[];
  public emotionalTriggers: EmotionalTrigger[];
  public manipulationVulnerabilities: ManipulationVulnerability[];
  public communicationPreferences: CommunicationPreference[];
  public decisionMakingPatterns: DecisionMakingPattern[];
  public socialDynamics: SocialDynamic[];
  public profileScore: number;
  public confidence: number;
  public lastUpdated: Date;

  constructor(
    targetId: string,
    personalityTraits: PersonalityTrait[] = [],
    cognitivePatterns: CognitivePattern[] = [],
    behavioralTendencies: BehavioralTendency[] = [],
    emotionalTriggers: EmotionalTrigger[] = [],
    manipulationVulnerabilities: ManipulationVulnerability[] = [],
    communicationPreferences: CommunicationPreference[] = [],
    decisionMakingPatterns: DecisionMakingPattern[] = [],
    socialDynamics: SocialDynamic[] = []
  ) {
    this.id = this.generateProfileId();
    this.targetId = targetId;
    this.timestamp = new Date();
    this.personalityTraits = personalityTraits;
    this.cognitivePatterns = cognitivePatterns;
    this.behavioralTendencies = behavioralTendencies;
    this.emotionalTriggers = emotionalTriggers;
    this.manipulationVulnerabilities = manipulationVulnerabilities;
    this.communicationPreferences = communicationPreferences;
    this.decisionMakingPatterns = decisionMakingPatterns;
    this.socialDynamics = socialDynamics;
    this.profileScore = this.calculateProfileScore();
    this.confidence = this.calculateConfidence();
    this.lastUpdated = new Date();
  }

  /**
   * Generate unique profile ID
   */
  private generateProfileId(): string {
    return `profile_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Calculate overall profile score based on completeness and confidence
   */
  private calculateProfileScore(): number {
    const weights = {
      personalityTraits: 0.2,
      cognitivePatterns: 0.15,
      behavioralTendencies: 0.15,
      emotionalTriggers: 0.1,
      manipulationVulnerabilities: 0.15,
      communicationPreferences: 0.1,
      decisionMakingPatterns: 0.1,
      socialDynamics: 0.05
    };

    let totalScore = 0;
    let totalWeight = 0;

    // Personality traits scoring
    if (this.personalityTraits.length > 0) {
      const avgTraitScore = this.personalityTraits.reduce((sum, trait) => sum + trait.score, 0) / this.personalityTraits.length;
      totalScore += avgTraitScore * weights.personalityTraits;
      totalWeight += weights.personalityTraits;
    }

    // Cognitive patterns scoring
    if (this.cognitivePatterns.length > 0) {
      const avgPatternScore = this.cognitivePatterns.reduce((sum, pattern) => sum + pattern.strength, 0) / this.cognitivePatterns.length;
      totalScore += avgPatternScore * weights.cognitivePatterns;
      totalWeight += weights.cognitivePatterns;
    }

    // Behavioral tendencies scoring
    if (this.behavioralTendencies.length > 0) {
      const avgTendencyScore = this.behavioralTendencies.reduce((sum, tendency) => sum + tendency.predictability, 0) / this.behavioralTendencies.length;
      totalScore += avgTendencyScore * weights.behavioralTendencies;
      totalWeight += weights.behavioralTendencies;
    }

    // Emotional triggers scoring
    if (this.emotionalTriggers.length > 0) {
      const avgTriggerScore = this.emotionalTriggers.reduce((sum, trigger) => sum + trigger.sensitivity, 0) / this.emotionalTriggers.length;
      totalScore += avgTriggerScore * weights.emotionalTriggers;
      totalWeight += weights.emotionalTriggers;
    }

    // Manipulation vulnerabilities scoring
    if (this.manipulationVulnerabilities.length > 0) {
      const avgVulnerabilityScore = this.manipulationVulnerabilities.reduce((sum, vuln) => sum + vuln.score, 0) / this.manipulationVulnerabilities.length;
      totalScore += avgVulnerabilityScore * weights.manipulationVulnerabilities;
      totalWeight += weights.manipulationVulnerabilities;
    }

    // Communication preferences scoring
    if (this.communicationPreferences.length > 0) {
      const avgCommScore = this.communicationPreferences.reduce((sum, pref) => sum + pref.effectiveness, 0) / this.communicationPreferences.length;
      totalScore += avgCommScore * weights.communicationPreferences;
      totalWeight += weights.communicationPreferences;
    }

    // Decision making patterns scoring
    if (this.decisionMakingPatterns.length > 0) {
      const avgDecisionScore = this.decisionMakingPatterns.reduce((sum, pattern) => {
        let score = 0;
        if (pattern.decisionSpeed === 'fast') score += 0.8;
        else if (pattern.decisionSpeed === 'moderate') score += 0.6;
        else score += 0.4;
        return sum + score;
      }, 0) / this.decisionMakingPatterns.length;
      totalScore += avgDecisionScore * weights.decisionMakingPatterns;
      totalWeight += weights.decisionMakingPatterns;
    }

    // Social dynamics scoring
    if (this.socialDynamics.length > 0) {
      const avgSocialScore = this.socialDynamics.reduce((sum, dynamic) => sum + dynamic.influenceLevel, 0) / this.socialDynamics.length;
      totalScore += avgSocialScore * weights.socialDynamics;
      totalWeight += weights.socialDynamics;
    }

    return totalWeight > 0 ? totalScore / totalWeight : 0;
  }

  /**
   * Calculate confidence score based on data quality and completeness
   */
  private calculateConfidence(): number {
    const completenessFactors = [
      this.personalityTraits.length > 0,
      this.cognitivePatterns.length > 0,
      this.behavioralTendencies.length > 0,
      this.emotionalTriggers.length > 0,
      this.manipulationVulnerabilities.length > 0,
      this.communicationPreferences.length > 0,
      this.decisionMakingPatterns.length > 0,
      this.socialDynamics.length > 0
    ];

    const completenessScore = completenessFactors.filter(Boolean).length / completenessFactors.length;

    // Factor in data freshness
    const daysSinceUpdate = (Date.now() - this.lastUpdated.getTime()) / (1000 * 60 * 60 * 24);
    const freshnessScore = Math.max(0, 1 - daysSinceUpdate / 30); // Decreases over 30 days

    return (completenessScore * 0.7) + (freshnessScore * 0.3);
  }

  /**
   * Update profile with new data
   */
  updateProfile(updates: Partial<IPsychologicalProfile>): void {
    if (updates.personalityTraits) {
      this.personalityTraits = updates.personalityTraits;
    }
    if (updates.cognitivePatterns) {
      this.cognitivePatterns = updates.cognitivePatterns;
    }
    if (updates.behavioralTendencies) {
      this.behavioralTendencies = updates.behavioralTendencies;
    }
    if (updates.emotionalTriggers) {
      this.emotionalTriggers = updates.emotionalTriggers;
    }
    if (updates.manipulationVulnerabilities) {
      this.manipulationVulnerabilities = updates.manipulationVulnerabilities;
    }
    if (updates.communicationPreferences) {
      this.communicationPreferences = updates.communicationPreferences;
    }
    if (updates.decisionMakingPatterns) {
      this.decisionMakingPatterns = updates.decisionMakingPatterns;
    }
    if (updates.socialDynamics) {
      this.socialDynamics = updates.socialDynamics;
    }

    this.profileScore = this.calculateProfileScore();
    this.confidence = this.calculateConfidence();
    this.lastUpdated = new Date();
  }

  /**
   * Get manipulation vulnerability score for specific technique
   */
  getVulnerabilityScore(technique: string): number {
    const vulnerability = this.manipulationVulnerabilities.find(v => v.type === technique);
    return vulnerability ? vulnerability.score : 0;
  }

  /**
   * Get most effective communication style
   */
  getMostEffectiveCommunicationStyle(): string {
    if (this.communicationPreferences.length === 0) {
      return 'direct';
    }

    const sorted = this.communicationPreferences.sort((a, b) => b.effectiveness - a.effectiveness);
    return sorted[0].style;
  }

  /**
   * Get dominant personality traits
   */
  getDominantPersonalityTraits(threshold: number = 0.7): PersonalityTrait[] {
    return this.personalityTraits.filter(trait => trait.score >= threshold);
  }

  /**
   * Get high-risk emotional triggers
   */
  getHighRiskTriggers(threshold: number = 0.8): EmotionalTrigger[] {
    return this.emotionalTriggers.filter(trigger => trigger.sensitivity >= threshold);
  }

  /**
   * Check if profile is complete enough for operations
   */
  isProfileComplete(minScore: number = 0.6): boolean {
    return this.profileScore >= minScore && this.confidence >= 0.5;
  }

  /**
   * Get profile summary for reporting
   */
  getProfileSummary(): Record<string, any> {
    return {
      id: this.id,
      targetId: this.targetId,
      profileScore: this.profileScore,
      confidence: this.confidence,
      lastUpdated: this.lastUpdated,
      dominantTraits: this.getDominantPersonalityTraits().map(t => t.name),
      highRiskTriggers: this.getHighRiskTriggers().map(t => t.type),
      primaryCommunicationStyle: this.getMostEffectiveCommunicationStyle(),
      manipulationVulnerabilities: this.manipulationVulnerabilities.map(v => ({
        type: v.type,
        score: v.score
      })),
      socialInfluenceLevel: this.socialDynamics.length > 0 ?
        this.socialDynamics.reduce((sum, dynamic) => sum + dynamic.influenceLevel, 0) / this.socialDynamics.length : 0
    };
  }

  /**
   * Clone profile for modification
   */
  clone(): PsychologicalProfile {
    return new PsychologicalProfile(
      this.targetId,
      [...this.personalityTraits],
      [...this.cognitivePatterns],
      [...this.behavioralTendencies],
      [...this.emotionalTriggers],
      [...this.manipulationVulnerabilities],
      [...this.communicationPreferences],
      [...this.decisionMakingPatterns],
      [...this.socialDynamics]
    );
  }

  /**
   * Merge with another profile
   */
  merge(other: PsychologicalProfile, weight: number = 0.5): PsychologicalProfile {
    const merged = this.clone();

    // Simple merge strategy - in production, this would be more sophisticated
    if (other.personalityTraits.length > 0) {
      other.personalityTraits.forEach(otherTrait => {
        const existingTrait = merged.personalityTraits.find(t => t.name === otherTrait.name);
        if (existingTrait) {
          existingTrait.score = (existingTrait.score * (1 - weight)) + (otherTrait.score * weight);
        } else {
          merged.personalityTraits.push({
            name: otherTrait.name,
            score: otherTrait.score * weight,
            description: otherTrait.description,
            manipulationImplications: otherTrait.manipulationImplications
          });
        }
      });
    }

    merged.profileScore = this.calculateProfileScore();
    merged.confidence = Math.min(this.confidence, other.confidence);
    merged.lastUpdated = new Date();

    return merged;
  }

  /**
   * Export profile for storage/serialization
   */
  export(): Record<string, any> {
    return {
      id: this.id,
      targetId: this.targetId,
      timestamp: this.timestamp,
      personalityTraits: this.personalityTraits,
      cognitivePatterns: this.cognitivePatterns,
      behavioralTendencies: this.behavioralTendencies,
      emotionalTriggers: this.emotionalTriggers,
      manipulationVulnerabilities: this.manipulationVulnerabilities,
      communicationPreferences: this.communicationPreferences,
      decisionMakingPatterns: this.decisionMakingPatterns,
      socialDynamics: this.socialDynamics,
      profileScore: this.profileScore,
      confidence: this.confidence,
      lastUpdated: this.lastUpdated
    };
  }

  /**
   * Import profile from storage/serialization
   */
  static import(data: Record<string, any>): PsychologicalProfile {
    const profile = new PsychologicalProfile(
      data.targetId,
      data.personalityTraits || [],
      data.cognitivePatterns || [],
      data.behavioralTendencies || [],
      data.emotionalTriggers || [],
      data.manipulationVulnerabilities || [],
      data.communicationPreferences || [],
      data.decisionMakingPatterns || [],
      data.socialDynamics || []
    );

    profile.id = data.id;
    profile.timestamp = new Date(data.timestamp);
    profile.profileScore = data.profileScore;
    profile.confidence = data.confidence;
    profile.lastUpdated = new Date(data.lastUpdated);

    return profile;
  }
}