// Enhanced User Journey and Business Analytics Monitoring
// Comprehensive frontend monitoring for user experience, conversion tracking,
// and business intelligence in the Chimera Next.js application

import { performanceMonitor } from './performance-monitor';

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface UserJourneyStep {
  stepId: string;
  stepName: string;
  stepType: 'page_view' | 'interaction' | 'form_submit' | 'api_call' | 'feature_use';
  timestamp: number;
  success: boolean;
  duration?: number;
  metadata?: Record<string, any>;
  errorDetails?: {
    type: string;
    message: string;
    stack?: string;
  };
}

export interface UserSession {
  sessionId: string;
  userId?: string;
  startTime: number;
  endTime?: number;
  totalDuration?: number;

  // Device and environment
  userAgent: string;
  viewport: { width: number; height: number };
  deviceType: 'mobile' | 'tablet' | 'desktop';
  connectionType: string;

  // User journey
  journeySteps: UserJourneyStep[];
  currentJourney?: string;
  journeyStartTime?: number;

  // Engagement metrics
  totalInteractions: number;
  featuresUsed: string[];
  pagesVisited: string[];
  timeOnPage: Record<string, number>;

  // Conversion metrics
  conversionEvents: ConversionEvent[];
  conversionValue?: number;

  // Quality of experience
  errorCount: number;
  satisfactionRating?: number; // 1-5 if provided by user
  npsScore?: number; // Net Promoter Score if collected
}

export interface ConversionEvent {
  eventId: string;
  eventType: 'signup' | 'subscription' | 'feature_adoption' | 'task_completion' | 'referral';
  timestamp: number;
  value?: number; // Monetary value if applicable
  metadata: Record<string, any>;
}

export interface FeatureUsageMetric {
  featureId: string;
  featureName: string;
  usageCount: number;
  successfulUses: number;
  averageTimeToComplete: number;
  abandonmentRate: number;
  userSatisfactionScore: number;
  lastUsed: number;
}

export interface UserSegment {
  segmentId: string;
  segmentName: string;
  criteria: Record<string, any>;
  userCount: number;
  averageEngagement: number;
  conversionRate: number;
  retentionRate: number;
}

export interface BusinessMetrics {
  timestamp: number;

  // User acquisition
  newUsers: number;
  returningUsers: number;
  totalSessions: number;

  // Engagement
  averageSessionDuration: number;
  pagesPerSession: number;
  bounceRate: number;

  // Conversion
  conversionRate: number;
  totalConversions: number;
  totalConversionValue: number;

  // Feature adoption
  featureAdoptionRates: Record<string, number>;
  featureUsageStats: FeatureUsageMetric[];

  // Quality metrics
  errorRate: number;
  averageLoadTime: number;
  coreWebVitalsScore: number;
  userSatisfactionScore: number;
}

// ============================================================================
// Journey Definition System
// ============================================================================

interface JourneyDefinition {
  journeyId: string;
  journeyName: string;
  description: string;
  expectedSteps: string[];
  maxDuration: number; // Maximum expected duration in ms
  criticalPath: boolean; // Is this a critical business journey?
  conversionGoal?: string; // What conversion event marks success?
}

class UserJourneyTracker {
  private journeyDefinitions: Map<string, JourneyDefinition> = new Map();
  private activeJourneys: Map<string, {
    journeyId: string;
    startTime: number;
    completedSteps: string[];
    sessionId: string;
  }> = new Map();

  constructor() {
    this.initializeJourneyDefinitions();
  }

  private initializeJourneyDefinitions() {
    // Define key user journeys for Chimera
    const journeys: JourneyDefinition[] = [
      {
        journeyId: 'onboarding',
        journeyName: 'User Onboarding',
        description: 'First-time user setup and initial prompt creation',
        expectedSteps: ['landing_page', 'signup', 'welcome', 'first_prompt', 'first_generation'],
        maxDuration: 300000, // 5 minutes
        criticalPath: true,
        conversionGoal: 'first_generation_success'
      },
      {
        journeyId: 'prompt_enhancement',
        journeyName: 'Prompt Enhancement Flow',
        description: 'User enhances a prompt using transformation techniques',
        expectedSteps: ['prompt_input', 'technique_selection', 'enhancement_preview', 'enhancement_apply', 'result_evaluation'],
        maxDuration: 120000, // 2 minutes
        criticalPath: true,
        conversionGoal: 'enhancement_success'
      },
      {
        journeyId: 'jailbreak_research',
        journeyName: 'Jailbreak Research Workflow',
        description: 'Advanced user conducts jailbreak research',
        expectedSteps: ['research_setup', 'technique_config', 'test_execution', 'results_analysis', 'report_generation'],
        maxDuration: 600000, // 10 minutes
        criticalPath: false,
        conversionGoal: 'research_completion'
      },
      {
        journeyId: 'provider_comparison',
        journeyName: 'LLM Provider Comparison',
        description: 'User compares responses across multiple LLM providers',
        expectedSteps: ['provider_selection', 'prompt_submission', 'response_generation', 'comparison_view', 'provider_preference'],
        maxDuration: 180000, // 3 minutes
        criticalPath: true,
        conversionGoal: 'comparison_completion'
      }
    ];

    journeys.forEach(journey => {
      this.journeyDefinitions.set(journey.journeyId, journey);
    });
  }

  startJourney(journeyId: string, sessionId: string): boolean {
    const journeyDef = this.journeyDefinitions.get(journeyId);
    if (!journeyDef) {
      console.warn(`Unknown journey: ${journeyId}`);
      return false;
    }

    // End any existing journey for this session
    this.endActiveJourney(sessionId);

    this.activeJourneys.set(sessionId, {
      journeyId,
      startTime: Date.now(),
      completedSteps: [],
      sessionId
    });

    this.sendJourneyEvent({
      type: 'journey_started',
      journeyId,
      sessionId,
      timestamp: Date.now()
    });

    return true;
  }

  recordJourneyStep(sessionId: string, stepId: string, success: boolean, metadata?: Record<string, any>): void {
    const activeJourney = this.activeJourneys.get(sessionId);
    if (!activeJourney) {
      // Auto-detect journey based on step
      const detectedJourney = this.detectJourneyFromStep(stepId);
      if (detectedJourney) {
        this.startJourney(detectedJourney, sessionId);
        this.recordJourneyStep(sessionId, stepId, success, metadata);
      }
      return;
    }

    const journeyDef = this.journeyDefinitions.get(activeJourney.journeyId);
    if (!journeyDef) return;

    // Record the step
    const stepData = {
      stepId,
      stepName: stepId,
      stepType: 'interaction' as const,
      timestamp: Date.now(),
      success,
      duration: Date.now() - activeJourney.startTime,
      metadata
    };

    if (success) {
      activeJourney.completedSteps.push(stepId);
    }

    // Send step event
    this.sendJourneyEvent({
      type: 'journey_step',
      journeyId: activeJourney.journeyId,
      sessionId,
      stepId,
      success,
      stepIndex: activeJourney.completedSteps.length,
      totalSteps: journeyDef.expectedSteps.length,
      timestamp: Date.now(),
      metadata
    });

    // Check for journey completion
    if (this.isJourneyComplete(activeJourney, journeyDef)) {
      this.completeJourney(sessionId, activeJourney);
    }

    // Check for journey timeout
    const journeyDuration = Date.now() - activeJourney.startTime;
    if (journeyDuration > journeyDef.maxDuration) {
      this.timeoutJourney(sessionId, activeJourney, 'timeout');
    }
  }

  private detectJourneyFromStep(stepId: string): string | null {
    // Map steps to likely journeys
    const stepJourneyMap: Record<string, string> = {
      'signup': 'onboarding',
      'welcome': 'onboarding',
      'first_prompt': 'onboarding',
      'technique_selection': 'prompt_enhancement',
      'enhancement_preview': 'prompt_enhancement',
      'provider_selection': 'provider_comparison',
      'research_setup': 'jailbreak_research'
    };

    return stepJourneyMap[stepId] || null;
  }

  private isJourneyComplete(activeJourney: any, journeyDef: JourneyDefinition): boolean {
    // Check if all critical steps are completed
    const criticalSteps = journeyDef.expectedSteps.slice(0, -1); // All but last are critical
    return criticalSteps.every(step => activeJourney.completedSteps.includes(step));
  }

  private completeJourney(sessionId: string, activeJourney: any): void {
    const duration = Date.now() - activeJourney.startTime;

    this.sendJourneyEvent({
      type: 'journey_completed',
      journeyId: activeJourney.journeyId,
      sessionId,
      duration,
      completedSteps: activeJourney.completedSteps.length,
      timestamp: Date.now()
    });

    this.activeJourneys.delete(sessionId);
  }

  private timeoutJourney(sessionId: string, activeJourney: any, reason: string): void {
    const duration = Date.now() - activeJourney.startTime;

    this.sendJourneyEvent({
      type: 'journey_abandoned',
      journeyId: activeJourney.journeyId,
      sessionId,
      duration,
      completedSteps: activeJourney.completedSteps.length,
      abandonment_reason: reason,
      timestamp: Date.now()
    });

    this.activeJourneys.delete(sessionId);
  }

  private endActiveJourney(sessionId: string): void {
    const activeJourney = this.activeJourneys.get(sessionId);
    if (activeJourney) {
      this.timeoutJourney(sessionId, activeJourney, 'new_journey_started');
    }
  }

  private sendJourneyEvent(event: any): void {
    // Send to backend analytics
    fetch('/api/analytics/journey-events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event)
    }).catch(console.error);
  }

  getActiveJourneys(): Record<string, any> {
    return Object.fromEntries(this.activeJourneys);
  }
}

// ============================================================================
// Feature Usage Tracking
// ============================================================================

class FeatureUsageTracker {
  private featureUsage: Map<string, FeatureUsageMetric> = new Map();
  private sessionFeatures: Set<string> = new Set();

  trackFeatureUsage(
    featureId: string,
    featureName: string,
    outcome: 'success' | 'error' | 'abandoned',
    duration: number = 0,
    metadata?: Record<string, any>
  ): void {
    const existing = this.featureUsage.get(featureId) || {
      featureId,
      featureName,
      usageCount: 0,
      successfulUses: 0,
      averageTimeToComplete: 0,
      abandonmentRate: 0,
      userSatisfactionScore: 0,
      lastUsed: 0
    };

    // Update metrics
    existing.usageCount++;
    existing.lastUsed = Date.now();

    if (outcome === 'success') {
      existing.successfulUses++;

      // Update average time to complete (running average)
      existing.averageTimeToComplete =
        (existing.averageTimeToComplete * (existing.successfulUses - 1) + duration) / existing.successfulUses;
    }

    // Calculate abandonment rate
    const abandonments = existing.usageCount - existing.successfulUses;
    existing.abandonmentRate = abandonments / existing.usageCount;

    this.featureUsage.set(featureId, existing);
    this.sessionFeatures.add(featureId);

    // Send analytics event
    this.sendFeatureEvent({
      type: 'feature_usage',
      featureId,
      featureName,
      outcome,
      duration,
      sessionId: performanceMonitor.getSessionMetrics().sessionId,
      timestamp: Date.now(),
      metadata
    });
  }

  recordFeatureFeedback(featureId: string, satisfactionScore: number): void {
    const existing = this.featureUsage.get(featureId);
    if (existing) {
      // Simple running average for satisfaction
      existing.userSatisfactionScore =
        (existing.userSatisfactionScore + satisfactionScore) / 2;

      this.featureUsage.set(featureId, existing);
    }
  }

  getFeatureUsageStats(): FeatureUsageMetric[] {
    return Array.from(this.featureUsage.values());
  }

  getSessionFeatures(): string[] {
    return Array.from(this.sessionFeatures);
  }

  private sendFeatureEvent(event: any): void {
    fetch('/api/analytics/feature-events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event)
    }).catch(console.error);
  }
}

// ============================================================================
// Conversion Tracking
// ============================================================================

class ConversionTracker {
  private conversions: ConversionEvent[] = [];
  private conversionGoals: Map<string, { value: number; priority: 'high' | 'medium' | 'low' }> = new Map();

  constructor() {
    this.initializeConversionGoals();
  }

  private initializeConversionGoals() {
    // Define conversion events and their business value
    this.conversionGoals.set('signup', { value: 10, priority: 'high' });
    this.conversionGoals.set('first_generation', { value: 25, priority: 'high' });
    this.conversionGoals.set('subscription', { value: 100, priority: 'high' });
    this.conversionGoals.set('feature_adoption', { value: 15, priority: 'medium' });
    this.conversionGoals.set('research_completion', { value: 50, priority: 'medium' });
    this.conversionGoals.set('referral', { value: 30, priority: 'low' });
  }

  trackConversion(
    eventType: ConversionEvent['eventType'],
    metadata: Record<string, any> = {},
    customValue?: number
  ): void {
    const goalInfo = this.conversionGoals.get(eventType);
    const value = customValue || goalInfo?.value || 0;

    const conversion: ConversionEvent = {
      eventId: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      eventType,
      timestamp: Date.now(),
      value,
      metadata: {
        ...metadata,
        priority: goalInfo?.priority || 'low',
        sessionId: performanceMonitor.getSessionMetrics().sessionId
      }
    };

    this.conversions.push(conversion);

    // Send conversion event
    this.sendConversionEvent(conversion);

    // Track in performance monitor
    performanceMonitor.trackUserInteraction(
      `conversion_${eventType}`,
      `conversion_tracker`,
      0
    );
  }

  getConversions(): ConversionEvent[] {
    return this.conversions;
  }

  getTotalConversionValue(): number {
    return this.conversions.reduce((total, conv) => total + (conv.value || 0), 0);
  }

  getConversionRate(eventType?: ConversionEvent['eventType']): number {
    const totalSessions = 1; // Would be calculated from actual session data
    const relevantConversions = eventType
      ? this.conversions.filter(c => c.eventType === eventType)
      : this.conversions;

    return relevantConversions.length / totalSessions;
  }

  private sendConversionEvent(conversion: ConversionEvent): void {
    fetch('/api/analytics/conversion-events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(conversion)
    }).catch(console.error);
  }
}

// ============================================================================
// Enhanced Business Analytics Monitor
// ============================================================================

class BusinessAnalyticsMonitor {
  private journeyTracker: UserJourneyTracker;
  private featureTracker: FeatureUsageTracker;
  private conversionTracker: ConversionTracker;
  private sessionStartTime: number;
  private errorCount: number = 0;

  constructor() {
    this.journeyTracker = new UserJourneyTracker();
    this.featureTracker = new FeatureUsageTracker();
    this.conversionTracker = new ConversionTracker();
    this.sessionStartTime = Date.now();

    this.initializeEventListeners();
  }

  private initializeEventListeners(): void {
    // Track errors
    window.addEventListener('error', (event) => {
      this.errorCount++;
      this.trackError('javascript_error', event.error?.message || 'Unknown error', {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.errorCount++;
      this.trackError('unhandled_promise_rejection', event.reason?.message || 'Promise rejected', {
        reason: event.reason
      });
    });

    // Track page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.flushAnalytics();
      }
    });

    // Track before unload
    window.addEventListener('beforeunload', () => {
      this.flushAnalytics();
    });
  }

  // Journey Management
  startJourney(journeyId: string): boolean {
    const sessionId = performanceMonitor.getSessionMetrics().sessionId;
    return this.journeyTracker.startJourney(journeyId, sessionId);
  }

  recordJourneyStep(stepId: string, success: boolean, metadata?: Record<string, any>): void {
    const sessionId = performanceMonitor.getSessionMetrics().sessionId;
    this.journeyTracker.recordJourneyStep(sessionId, stepId, success, metadata);
  }

  // Feature Usage
  trackFeatureUsage(
    featureId: string,
    featureName: string,
    outcome: 'success' | 'error' | 'abandoned',
    duration: number = 0,
    metadata?: Record<string, any>
  ): void {
    this.featureTracker.trackFeatureUsage(featureId, featureName, outcome, duration, metadata);
  }

  recordFeatureFeedback(featureId: string, satisfactionScore: number): void {
    this.featureTracker.recordFeatureFeedback(featureId, satisfactionScore);
  }

  // Conversion Tracking
  trackConversion(
    eventType: ConversionEvent['eventType'],
    metadata?: Record<string, any>,
    customValue?: number
  ): void {
    this.conversionTracker.trackConversion(eventType, metadata, customValue);
  }

  // Error Tracking
  trackError(errorType: string, message: string, metadata?: Record<string, any>): void {
    const errorEvent = {
      type: 'application_error',
      errorType,
      message,
      timestamp: Date.now(),
      sessionId: performanceMonitor.getSessionMetrics().sessionId,
      url: window.location.href,
      userAgent: navigator.userAgent,
      metadata
    };

    fetch('/api/analytics/error-events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorEvent)
    }).catch(console.error);
  }

  // Business Metrics
  getBusinessMetrics(): BusinessMetrics {
    const sessionMetrics = performanceMonitor.getSessionMetrics();
    const performanceSummary = performanceMonitor.getPerformanceSummary();

    return {
      timestamp: Date.now(),

      // User acquisition (would be calculated from actual data)
      newUsers: 1, // Placeholder
      returningUsers: 0, // Placeholder
      totalSessions: 1,

      // Engagement
      averageSessionDuration: Date.now() - this.sessionStartTime,
      pagesPerSession: 1, // Would track page visits
      bounceRate: 0, // Would calculate from session data

      // Conversion
      conversionRate: this.conversionTracker.getConversionRate(),
      totalConversions: this.conversionTracker.getConversions().length,
      totalConversionValue: this.conversionTracker.getTotalConversionValue(),

      // Feature adoption
      featureAdoptionRates: this.calculateFeatureAdoptionRates(),
      featureUsageStats: this.featureTracker.getFeatureUsageStats(),

      // Quality metrics
      errorRate: this.errorCount / Math.max(sessionMetrics.apiMetrics?.length || 1, 1),
      averageLoadTime: performanceSummary.coreVitals?.LCP?.value || 0,
      coreWebVitalsScore: this.calculateCoreWebVitalsScore(performanceSummary),
      userSatisfactionScore: this.calculateUserSatisfactionScore()
    };
  }

  private calculateFeatureAdoptionRates(): Record<string, number> {
    const features = this.featureTracker.getFeatureUsageStats();
    const adoptionRates: Record<string, number> = {};

    features.forEach(feature => {
      // Simple adoption rate calculation (could be more sophisticated)
      adoptionRates[feature.featureId] = feature.successfulUses / Math.max(feature.usageCount, 1);
    });

    return adoptionRates;
  }

  private calculateCoreWebVitalsScore(summary: any): number {
    const vitals = summary.coreVitals;
    if (!vitals) return 0;

    let score = 0;
    let count = 0;

    // Score each vital (good=100, needs improvement=50, poor=0)
    Object.values(vitals).forEach((vital: any) => {
      if (vital?.rating) {
        switch (vital.rating) {
          case 'good': score += 100; break;
          case 'needs-improvement': score += 50; break;
          case 'poor': score += 0; break;
        }
        count++;
      }
    });

    return count > 0 ? score / count : 0;
  }

  private calculateUserSatisfactionScore(): number {
    const features = this.featureTracker.getFeatureUsageStats();
    if (features.length === 0) return 0;

    const totalSatisfaction = features.reduce(
      (sum, feature) => sum + feature.userSatisfactionScore,
      0
    );

    return totalSatisfaction / features.length;
  }

  private flushAnalytics(): void {
    const businessMetrics = this.getBusinessMetrics();

    // Send final analytics data
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        '/api/analytics/session-summary',
        JSON.stringify(businessMetrics)
      );
    }
  }

  // Public API for easy integration
  getAnalyticsSummary() {
    return {
      activeJourneys: this.journeyTracker.getActiveJourneys(),
      featureUsage: this.featureTracker.getFeatureUsageStats(),
      conversions: this.conversionTracker.getConversions(),
      businessMetrics: this.getBusinessMetrics()
    };
  }
}

// ============================================================================
// React Hooks for Easy Integration
// ============================================================================

export const useJourneyTracking = (journeyId?: string) => {
  const startJourney = (id: string = journeyId!) => businessAnalytics.startJourney(id);
  const recordStep = (stepId: string, success: boolean = true, metadata?: Record<string, any>) =>
    businessAnalytics.recordJourneyStep(stepId, success, metadata);

  return { startJourney, recordStep };
};

export const useFeatureTracking = (featureId: string, featureName: string) => {
  const trackUsage = (outcome: 'success' | 'error' | 'abandoned', duration?: number, metadata?: Record<string, any>) =>
    businessAnalytics.trackFeatureUsage(featureId, featureName, outcome, duration, metadata);

  const recordFeedback = (score: number) => businessAnalytics.recordFeatureFeedback(featureId, score);

  return { trackUsage, recordFeedback };
};

export const useConversionTracking = () => {
  const trackConversion = (eventType: ConversionEvent['eventType'], metadata?: Record<string, any>, value?: number) =>
    businessAnalytics.trackConversion(eventType, metadata, value);

  return { trackConversion };
};

export const useErrorTracking = () => {
  const trackError = (errorType: string, message: string, metadata?: Record<string, any>) =>
    businessAnalytics.trackError(errorType, message, metadata);

  return { trackError };
};

// ============================================================================
// Global Instance and Exports
// ============================================================================

export const businessAnalytics = new BusinessAnalyticsMonitor();

// Auto-start onboarding journey for new users
if (typeof window !== 'undefined') {
  // Check if this looks like a new user (could be more sophisticated)
  const isNewUser = !localStorage.getItem('chimera_returning_user');
  if (isNewUser) {
    setTimeout(() => businessAnalytics.startJourney('onboarding'), 1000);
    localStorage.setItem('chimera_returning_user', 'true');
  }
}

export default businessAnalytics;
