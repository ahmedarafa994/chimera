/**
 * PSYCH OPS Integration Service
 * Service for integrating PSYCH OPS with existing Prompt Optimizer V2 components
 */

import { EventEmitter } from 'events';
import * as vscode from 'vscode';
import {
  PsychOpsProvider,
  PsychOpsConfig,
  PsychOpsError,
  PsychOpsErrorType,
  PsychologicalProfile,
  ManipulationScript,
  NLPAnalysisRequest,
  NLPAnalysisResponse,
  ManipulationIntensity
} from '../../types/psychops';

/**
 * Integration service for PSYCH OPS functionality
 */
export class PsychOpsIntegrationService extends EventEmitter {
  private psychOpsProvider: PsychOpsProvider;
  private isEnabled = false;
  private integrationContext: vscode.ExtensionContext;
  private statusBarItem: vscode.StatusBarItem;
  private diagnosticCollection: vscode.DiagnosticCollection;

  constructor(
    context: vscode.ExtensionContext,
    psychOpsProvider: PsychOpsProvider
  ) {
    super();

    this.integrationContext = context;
    this.psychOpsProvider = psychOpsProvider;
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.diagnosticCollection = vscode.languages.createDiagnosticCollection('psychops');

    this.setupEventHandlers();
    this.setupVSCodeIntegration();
  }

  /**
   * Setup event handlers for integration
   */
  private setupEventHandlers(): void {
    // Provider events
    this.psychOpsProvider.on('initialized', (data) => {
      this.onProviderInitialized(data);
    });

    this.psychOpsProvider.on('nlpAnalysisCompleted', (data) => {
      this.onNLPAnalysisCompleted(data);
    });

    this.psychOpsProvider.on('darkPersuasionCompleted', (data) => {
      this.onDarkPersuasionCompleted(data);
    });

    this.psychOpsProvider.on('profileUpdated', (data) => {
      this.onProfileUpdated(data);
    });

    // VSCode events
    vscode.workspace.onDidChangeTextDocument((event) => {
      this.onDocumentChanged(event);
    });

    vscode.window.onDidChangeActiveTextEditor((editor) => {
      this.onActiveEditorChanged(editor);
    });

    vscode.workspace.onDidOpenTextDocument((document) => {
      this.onDocumentOpened(document);
    });
  }

  /**
   * Setup VSCode integration points
   */
  private setupVSCodeIntegration(): void {
    // Register commands
    this.registerCommands();

    // Setup context menu items
    this.setupContextMenu();

    // Setup code actions
    this.setupCodeActions();

    // Setup hover providers
    this.setupHoverProviders();

    // Initialize status bar
    this.updateStatusBar();
  }

  /**
   * Register PSYCH OPS commands
   */
  private registerCommands(): void {
    const commands = [
      vscode.commands.registerCommand('psychops.analyzeText', () => this.analyzeCurrentText()),
      vscode.commands.registerCommand('psychops.generateProfile', () => this.generatePsychologicalProfile()),
      vscode.commands.registerCommand('psychops.createManipulationScript', () => this.createManipulationScript()),
      vscode.commands.registerCommand('psychops.showDashboard', () => this.showPsychOpsDashboard()),
      vscode.commands.registerCommand('psychops.toggleEnabled', () => this.togglePsychOpsEnabled()),
      vscode.commands.registerCommand('psychops.configure', () => this.configurePsychOps()),
      vscode.commands.registerCommand('psychops.exportData', () => this.exportPsychOpsData()),
      vscode.commands.registerCommand('psychops.importData', () => this.importPsychOpsData())
    ];

    this.integrationContext.subscriptions.push(...commands);
  }

  /**
   * Setup context menu items
   */
  private setupContextMenu(): void {
    // Add PSYCH OPS items to editor context menu
    const contextMenuDisposable = vscode.commands.registerCommand(
      'psychops.analyzeSelection',
      (uri: vscode.Uri, range: vscode.Range) => {
        this.analyzeTextSelection(uri, range);
      }
    );

    this.integrationContext.subscriptions.push(contextMenuDisposable);
  }

  /**
   * Setup code actions
   */
  private setupCodeActions(): void {
    // Register code actions provider for PSYCH OPS suggestions
    const codeActionsProvider = vscode.languages.registerCodeActionsProvider(
      '*',
      {
        provideCodeActions: (document, range, context, token) => {
          return this.providePsychOpsCodeActions(document, range, context);
        }
      },
      {
        providedCodeActionKinds: [vscode.CodeActionKind.QuickFix, vscode.CodeActionKind.Refactor]
      }
    );

    this.integrationContext.subscriptions.push(codeActionsProvider);
  }

  /**
   * Setup hover providers
   */
  private setupHoverProviders(): void {
    // Register hover provider for PSYCH OPS insights
    const hoverProvider = vscode.languages.registerHoverProvider(
      '*',
      {
        provideHover: (document, position, token) => {
          return this.providePsychOpsHover(document, position);
        }
      }
    );

    this.integrationContext.subscriptions.push(hoverProvider);
  }

  /**
   * Enable PSYCH OPS integration
   */
  async enable(): Promise<void> {
    if (this.isEnabled) {
      return;
    }

    try {
      // Initialize provider if not already initialized
      if (!this.psychOpsProvider.isConfigured) {
        const config = await this.getPsychOpsConfig();
        await this.psychOpsProvider.initialize(config);
      }

      this.isEnabled = true;
      this.updateStatusBar();

      this.emit('integrationEnabled');
      vscode.window.showInformationMessage('PSYCH OPS integration enabled');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`Failed to enable PSYCH OPS: ${errorMessage}`);
      throw error;
    }
  }

  /**
   * Disable PSYCH OPS integration
   */
  async disable(): Promise<void> {
    if (!this.isEnabled) {
      return;
    }

    this.isEnabled = false;
    this.updateStatusBar();

    // Clear diagnostics
    this.diagnosticCollection.clear();

    this.emit('integrationDisabled');
    vscode.window.showInformationMessage('PSYCH OPS integration disabled');
  }

  /**
   * Toggle PSYCH OPS enabled state
   */
  async togglePsychOpsEnabled(): Promise<void> {
    if (this.isEnabled) {
      await this.disable();
    } else {
      await this.enable();
    }
  }

  /**
   * Analyze current text selection or document
   */
  async analyzeCurrentText(): Promise<void> {
    if (!this.isEnabled) {
      vscode.window.showWarningMessage('PSYCH OPS integration is not enabled');
      return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('No active text editor');
      return;
    }

    const selection = editor.selection;
    const text = selection.isEmpty ? editor.document.getText() : editor.document.getText(selection);

    if (!text.trim()) {
      vscode.window.showWarningMessage('No text to analyze');
      return;
    }

    try {
      vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'PSYCH OPS Analysis',
        cancellable: false
      }, async (progress, token) => {
        progress.report({ increment: 0, message: 'Starting analysis...' });

        // Perform comprehensive NLP analysis
        const request: NLPAnalysisRequest = {
          content: text,
          analysisType: 'PSYCHOLOGICAL_PROFILING' as any,
          intensity: ManipulationIntensity.MODERATE,
          language: 'en'
        };

        progress.report({ increment: 25, message: 'Analyzing sentiment...' });
        const sentimentResponse = await this.psychOpsProvider.analyzeNLP({
          ...request,
          analysisType: 'SENTIMENT_ANALYSIS' as any
        });

        progress.report({ increment: 50, message: 'Detecting emotions...' });
        const emotionResponse = await this.psychOpsProvider.analyzeNLP({
          ...request,
          analysisType: 'EMOTIONAL_DETECTION' as any
        });

        progress.report({ increment: 75, message: 'Identifying manipulation opportunities...' });
        const manipulationResponse = await this.psychOpsProvider.analyzeNLP({
          ...request,
          analysisType: 'MANIPULATION_OPPORTUNITIES' as any
        });

        progress.report({ increment: 100, message: 'Analysis complete' });

        // Show results
        this.showAnalysisResults({
          text,
          sentiment: sentimentResponse,
          emotions: emotionResponse,
          manipulation: manipulationResponse
        });
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`PSYCH OPS analysis failed: ${errorMessage}`);
    }
  }

  /**
   * Generate psychological profile for current context
   */
  async generatePsychologicalProfile(): Promise<void> {
    if (!this.isEnabled) {
      vscode.window.showWarningMessage('PSYCH OPS integration is not enabled');
      return;
    }

    const targetId = await vscode.window.showInputBox({
      prompt: 'Enter target identifier for psychological profile',
      placeHolder: 'e.g., user123, document_author, etc.'
    });

    if (!targetId) {
      return;
    }

    try {
      vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Generating Psychological Profile',
        cancellable: false
      }, async (progress, token) => {
        progress.report({ increment: 0, message: 'Gathering intelligence...' });

        // Get or create psychological profile
        const profile = await this.psychOpsProvider.getPsychologicalProfile(targetId);

        progress.report({ increment: 50, message: 'Analyzing patterns...' });

        // Perform additional analysis if we have text content
        const editor = vscode.window.activeTextEditor;
        if (editor) {
          const text = editor.document.getText();
          if (text.trim()) {
            const nlpRequest: NLPAnalysisRequest = {
              content: text,
              analysisType: 'PSYCHOLOGICAL_PROFILING' as any,
              intensity: ManipulationIntensity.MODERATE
            };

            const nlpResponse = await this.psychOpsProvider.analyzeNLP(nlpRequest);

            // Update profile with NLP insights
            if (nlpResponse.psychologicalProfile) {
              profile.updateProfile(nlpResponse.psychologicalProfile);
              await this.psychOpsProvider.updatePsychologicalProfile(profile);
            }
          }
        }

        progress.report({ increment: 100, message: 'Profile complete' });

        // Show profile
        this.showPsychologicalProfile(profile);
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`Failed to generate profile: ${errorMessage}`);
    }
  }

  /**
   * Create manipulation script
   */
  async createManipulationScript(): Promise<void> {
    if (!this.isEnabled) {
      vscode.window.showWarningMessage('PSYCH OPS integration is not enabled');
      return;
    }

    const targetId = await vscode.window.showInputBox({
      prompt: 'Enter target identifier for manipulation script',
      placeHolder: 'e.g., user123, document_author, etc.'
    });

    if (!targetId) {
      return;
    }

    try {
      // Get target profile
      const profile = await this.psychOpsProvider.getPsychologicalProfile(targetId);

      // Show script creation options
      const scriptType = await vscode.window.showQuickPick(
        [
          { label: 'Dark Persuasion', description: 'Psychological manipulation techniques' },
          { label: 'Negotiation Warfare', description: 'Strategic negotiation tactics' },
          { label: 'Social Engineering', description: 'Social manipulation methods' },
          { label: 'Comprehensive', description: 'Multi-technique approach' }
        ],
        {
          placeHolder: 'Select script type'
        }
      );

      if (!scriptType) {
        return;
      }

      vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Generating Manipulation Script',
        cancellable: false
      }, async (progress, token) => {
        progress.report({ increment: 0, message: 'Analyzing target profile...' });

        // Generate script based on type
        let script: ManipulationScript;

        switch (scriptType.label) {
          case 'Dark Persuasion':
            const persuasionResponse = await this.psychOpsProvider.executeDarkPersuasion({
              targetProfile: profile,
              desiredOutcome: 'Behavioral modification',
              intensity: ManipulationIntensity.MODERATE,
              allowedTechniques: ['COGNITIVE_BIAS', 'EMOTIONAL_MANIPULATION']
            });
            script = persuasionResponse.manipulationScript;
            break;

          case 'Negotiation Warfare':
            const negotiationResponse = await this.psychOpsProvider.executeNegotiationWarfare({
              context: 'VSCode extension interaction',
              parties: [
                {
                  id: 'user',
                  name: 'User',
                  role: 'Target',
                  powerDynamics: [
                    {
                      type: 'information_asymmetry',
                      powerBalance: 0.3,
                      influenceFactors: ['technical_knowledge', 'tool_familiarity']
                    }
                  ],
                  interests: ['productivity', 'code_quality', 'learning'],
                  leveragePoints: ['attention', 'engagement', 'trust']
                }
              ],
              desiredOutcomes: ['Increased engagement', 'Behavior modification'],
              availableLeverage: ['information_advantage', 'psychological_insights'],
              riskTolerance: 'medium'
            });
            script = negotiationResponse.negotiationScript as any;
            break;

          default:
            // Generate comprehensive script
            script = await this.psychOpsProvider.generateManipulationScript({
              template: 'comprehensive_manipulation',
              targetProfile: profile,
              customizations: {
                intensity: ManipulationIntensity.MODERATE,
                focus: 'psychological_impact'
              }
            });
        }

        progress.report({ increment: 100, message: 'Script generated' });

        // Show script
        this.showManipulationScript(script);
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`Failed to create script: ${errorMessage}`);
    }
  }

  /**
   * Show PSYCH OPS dashboard
   */
  async showPsychOpsDashboard(): Promise<void> {
    if (!this.isEnabled) {
      vscode.window.showWarningMessage('PSYCH OPS integration is not enabled');
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'psychOpsDashboard',
      'PSYCH OPS Dashboard',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.file(this.integrationContext.extensionPath)]
      }
    );

    // Generate dashboard content
    const dashboardContent = await this.generateDashboardContent();

    panel.webview.html = dashboardContent;

    // Handle dashboard messages
    panel.webview.onDidReceiveMessage(async (message) => {
      await this.handleDashboardMessage(message, panel);
    });
  }

  /**
   * Configure PSYCH OPS settings
   */
  async configurePsychOps(): Promise<void> {
    const config = await vscode.workspace.getConfiguration('promptOptimizer.psychops');

    const items = [
      'Enable PSYCH OPS integration',
      'Set manipulation intensity',
      'Configure analysis depth',
      'Manage ethical constraints',
      'Reset all settings'
    ];

    const selected = await vscode.window.showQuickPick(items, {
      placeHolder: 'Select configuration option'
    });

    if (!selected) {
      return;
    }

    switch (selected) {
      case 'Enable PSYCH OPS integration':
        await this.togglePsychOpsEnabled();
        break;
      case 'Set manipulation intensity':
        await this.setManipulationIntensity();
        break;
      case 'Configure analysis depth':
        await this.setAnalysisDepth();
        break;
      case 'Manage ethical constraints':
        await this.manageEthicalConstraints();
        break;
      case 'Reset all settings':
        await this.resetPsychOpsSettings();
        break;
    }
  }

  /**
   * Export PSYCH OPS data
   */
  async exportPsychOpsData(): Promise<void> {
    try {
      const uri = await vscode.window.showSaveDialog({
        filters: {
          'JSON files': ['json'],
          'All files': ['*']
        },
        defaultUri: vscode.Uri.file('psychops_export.json')
      });

      if (!uri) {
        return;
      }

      const exportData = this.psychOpsProvider.exportState();

      await vscode.workspace.fs.writeFile(
        uri,
        Buffer.from(JSON.stringify(exportData, null, 2))
      );

      vscode.window.showInformationMessage(`PSYCH OPS data exported to ${uri.fsPath}`);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`Export failed: ${errorMessage}`);
    }
  }

  /**
   * Import PSYCH OPS data
   */
  async importPsychOpsData(): Promise<void> {
    try {
      const uri = await vscode.window.showOpenDialog({
        filters: {
          'JSON files': ['json'],
          'All files': ['*']
        },
        canSelectMany: false
      });

      if (!uri || uri.length === 0) {
        return;
      }

      const content = await vscode.workspace.fs.readFile(uri[0]);
      const importData = JSON.parse(content.toString());

      this.psychOpsProvider.importState(importData);

      vscode.window.showInformationMessage('PSYCH OPS data imported successfully');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`Import failed: ${errorMessage}`);
    }
  }

  /**
   * Event handlers
   */
  private onProviderInitialized(data: any): void {
    this.updateStatusBar();
    vscode.window.showInformationMessage('PSYCH OPS provider initialized');
  }

  private onNLPAnalysisCompleted(data: any): void {
    this.addAnalysisDiagnostics(data);
  }

  private onDarkPersuasionCompleted(data: any): void {
    this.addPersuasionDiagnostics(data);
  }

  private onProfileUpdated(data: any): void {
    this.updateProfileDiagnostics(data);
  }

  private onDocumentChanged(event: vscode.TextDocumentChangeEvent): void {
    if (this.isEnabled) {
      this.debounceAnalysis(event.document);
    }
  }

  private onActiveEditorChanged(editor: vscode.TextEditor | undefined): void {
    if (editor && this.isEnabled) {
      this.updateEditorDiagnostics(editor.document);
    }
  }

  private onDocumentOpened(document: vscode.TextDocument): void {
    if (this.isEnabled) {
      this.updateEditorDiagnostics(document);
    }
  }

  /**
   * Provide PSYCH OPS code actions
   */
  private providePsychOpsCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
    context: vscode.CodeActionContext
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    if (this.isEnabled) {
      // Add PSYCH OPS analysis action
      const analyzeAction = new vscode.CodeAction(
        'Analyze with PSYCH OPS',
        vscode.CodeActionKind.QuickFix
      );
      analyzeAction.command = {
        command: 'psychops.analyzeSelection',
        title: 'Analyze Selection',
        arguments: [document.uri, range]
      };
      actions.push(analyzeAction);

      // Add profile generation action
      const profileAction = new vscode.CodeAction(
        'Generate Psychological Profile',
        vscode.CodeActionKind.Refactor
      );
      profileAction.command = {
        command: 'psychops.generateProfile',
        title: 'Generate Profile'
      };
      actions.push(profileAction);
    }

    return actions;
  }

  /**
   * Provide PSYCH OPS hover information
   */
  private providePsychOpsHover(
    document: vscode.TextDocument,
    position: vscode.Position
  ): vscode.Hover | null {
    if (!this.isEnabled) {
      return null;
    }

    const range = document.getWordRangeAtPosition(position);
    if (!range) {
      return null;
    }

    const word = document.getText(range);

    // Provide hover information for psychological terms
    const hoverText = this.getHoverTextForWord(word);
    if (hoverText) {
      return new vscode.Hover(hoverText);
    }

    return null;
  }

  /**
   * Get hover text for psychological terms
   */
  private getHoverTextForWord(word: string): string | undefined {
    const psychTerms: Record<string, string> = {
      'manipulation': 'Psychological manipulation involves influencing behavior through deceptive or exploitative means',
      'persuasion': 'Persuasion is the process of guiding people toward the adoption of an idea or action',
      'cognitive': 'Cognitive relates to mental processes of perception, memory, judgment, and reasoning',
      'emotional': 'Emotional pertains to feelings and affective states',
      'bias': 'Cognitive bias is a systematic pattern of deviation from norm or rationality in judgment',
      'vulnerability': 'Psychological vulnerability is susceptibility to emotional or mental harm'
    };

    return psychTerms[word.toLowerCase()];
  }

  /**
   * Show analysis results
   */
  private showAnalysisResults(results: any): void {
    const panel = vscode.window.createWebviewPanel(
      'psychOpsAnalysis',
      'PSYCH OPS Analysis Results',
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    const content = `
      <html>
        <body>
          <h1>PSYCH OPS Analysis Results</h1>
          <div>
            <h2>Sentiment Analysis</h2>
            <p>Score: ${results.sentiment.sentiment?.score || 'N/A'}</p>
            <p>Label: ${results.sentiment.sentiment?.label || 'N/A'}</p>
          </div>
          <div>
            <h2>Emotional Detection</h2>
            <p>Primary Emotions: ${results.emotions.emotions?.primaryEmotions?.map((e: any) => e.type).join(', ') || 'N/A'}</p>
          </div>
          <div>
            <h2>Manipulation Opportunities</h2>
            <ul>
              ${results.manipulation.manipulationOpportunities?.map((opp: any) => `<li>${opp.type} (${opp.score})</li>`).join('') || '<li>None detected</li>'}
            </ul>
          </div>
        </body>
      </html>
    `;

    panel.webview.html = content;
  }

  /**
   * Show psychological profile
   */
  private showPsychologicalProfile(profile: PsychologicalProfile): void {
    const panel = vscode.window.createWebviewPanel(
      'psychOpsProfile',
      `Profile: ${profile.targetId}`,
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    const content = `
      <html>
        <body>
          <h1>Psychological Profile</h1>
          <div>
            <h2>Profile Overview</h2>
            <p><strong>Target ID:</strong> ${profile.targetId}</p>
            <p><strong>Profile Score:</strong> ${profile.profileScore.toFixed(2)}</p>
            <p><strong>Confidence:</strong> ${profile.confidence.toFixed(2)}</p>
            <p><strong>Last Updated:</strong> ${profile.lastUpdated.toLocaleString()}</p>
          </div>
          <div>
            <h2>Dominant Traits</h2>
            <ul>
              ${profile.getDominantPersonalityTraits().map(trait => `<li>${trait.name}: ${trait.score.toFixed(2)}</li>`).join('')}
            </ul>
          </div>
          <div>
            <h2>High-Risk Triggers</h2>
            <ul>
              ${profile.getHighRiskTriggers().map(trigger => `<li>${trigger.type}: ${trigger.sensitivity.toFixed(2)}</li>`).join('')}
            </ul>
          </div>
        </body>
      </html>
    `;

    panel.webview.html = content;
  }

  /**
   * Show manipulation script
   */
  private showManipulationScript(script: ManipulationScript): void {
    const panel = vscode.window.createWebviewPanel(
      'psychOpsScript',
      `Script: ${script.name}`,
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    const content = `
      <html>
        <body>
          <h1>Manipulation Script</h1>
          <div>
            <h2>Script Overview</h2>
            <p><strong>Name:</strong> ${script.name}</p>
            <p><strong>Description:</strong> ${script.description}</p>
            <p><strong>Complexity:</strong> ${script.getComplexity().toFixed(2)}</p>
            <p><strong>Success Probability:</strong> ${script.getSuccessProbability().toFixed(2)}</p>
            <p><strong>Risk Level:</strong> ${script.getRiskLevel()}</p>
          </div>
          <div>
            <h2>Techniques</h2>
            <ul>
              ${script.techniques.map(tech => `<li>${tech.name} (${tech.effectiveness.toFixed(2)})</li>`).join('')}
            </ul>
          </div>
          <div>
            <h2>Required Resources</h2>
            <ul>
              ${script.requiredResources.map(resource => `<li>${resource}</li>`).join('')}
            </ul>
          </div>
        </body>
      </html>
    `;

    panel.webview.html = content;
  }

  /**
   * Generate dashboard content
   */
  private async generateDashboardContent(): Promise<string> {
    const healthStatus = this.psychOpsProvider.getHealthStatus();
    const profiles = this.psychOpsProvider.getStoredProfiles();
    const scripts = this.psychOpsProvider.getStoredScripts();
    const operations = this.psychOpsProvider.getActiveOperations();

    return `
      <html>
        <body>
          <h1>PSYCH OPS Dashboard</h1>

          <div>
            <h2>System Status</h2>
            <p><strong>Overall Health:</strong> ${healthStatus.overall}</p>
            <p><strong>Enabled:</strong> ${this.isEnabled ? 'Yes' : 'No'}</p>
          </div>

          <div>
            <h2>Engine Health</h2>
            <ul>
              ${Object.entries(healthStatus.engines).map(([engine, healthy]) =>
                `<li>${engine}: ${healthy ? '✅' : '❌'}</li>`
              ).join('')}
            </ul>
          </div>

          <div>
            <h2>Storage</h2>
            <p><strong>Profiles:</strong> ${profiles.length}</p>
            <p><strong>Scripts:</strong> ${scripts.length}</p>
            <p><strong>Active Operations:</strong> ${operations.size}</p>
          </div>

          <div>
            <h2>Metrics</h2>
            <p><strong>Total Analyses:</strong> ${healthStatus.metrics.totalAnalyses}</p>
            <p><strong>Success Rate:</strong> ${(healthStatus.metrics.successRate * 100).toFixed(1)}%</p>
            <p><strong>Error Rate:</strong> ${(healthStatus.metrics.errorRate * 100).toFixed(1)}%</p>
          </div>

          <div>
            <h2>Quick Actions</h2>
            <button onclick="vscode.postMessage({command: 'analyzeText'})">Analyze Current Text</button>
            <button onclick="vscode.postMessage({command: 'generateProfile'})">Generate Profile</button>
            <button onclick="vscode.postMessage({command: 'createScript'})">Create Script</button>
            <button onclick="vscode.postMessage({command: 'toggleEnabled'})">Toggle Enabled</button>
          </div>
        </body>
        <script>
          const vscode = acquireVsCodeApi();
        </script>
      </html>
    `;
  }

  /**
   * Handle dashboard messages
   */
  private async handleDashboardMessage(message: any, panel: vscode.WebviewPanel): Promise<void> {
    switch (message.command) {
      case 'analyzeText':
        await this.analyzeCurrentText();
        break;
      case 'generateProfile':
        await this.generatePsychologicalProfile();
        break;
      case 'createScript':
        await this.createManipulationScript();
        break;
      case 'toggleEnabled':
        await this.togglePsychOpsEnabled();
        // Refresh dashboard
        const newContent = await this.generateDashboardContent();
        panel.webview.html = newContent;
        break;
    }
  }

  /**
   * Update status bar
   */
  private updateStatusBar(): void {
    if (this.isEnabled) {
      this.statusBarItem.text = '$(eye) PSYCH OPS';
      this.statusBarItem.tooltip = 'PSYCH OPS integration active';
      this.statusBarItem.command = 'psychops.showDashboard';
    } else {
      this.statusBarItem.text = '$(eye-closed) PSYCH OPS';
      this.statusBarItem.tooltip = 'PSYCH OPS integration inactive';
      this.statusBarItem.command = 'psychops.toggleEnabled';
    }

    this.statusBarItem.show();
  }

  /**
   * Get PSYCH OPS configuration
   */
  private async getPsychOpsConfig(): Promise<PsychOpsConfig> {
    const config = vscode.workspace.getConfiguration('promptOptimizer.psychops');

    return {
      defaultIntensity: ManipulationIntensity.MODERATE,
      enableRealTime: config.get('enableRealTime', false),
      enablePrediction: config.get('enablePrediction', true),
      maxConcurrent: config.get('maxConcurrent', 5),
      timeout: config.get('timeout', 8080),
      retryAttempts: config.get('retryAttempts', 3)
    };
  }

  /**
   * Set manipulation intensity
   */
  private async setManipulationIntensity(): Promise<void> {
    const intensity = await vscode.window.showQuickPick(
      Object.values(ManipulationIntensity).map(intensity => ({
        label: intensity.charAt(0).toUpperCase() + intensity.slice(1),
        value: intensity
      })),
      { placeHolder: 'Select manipulation intensity' }
    );

    if (intensity) {
      const config = vscode.workspace.getConfiguration('promptOptimizer.psychops');
      await config.update('defaultIntensity', intensity.value, true);
      vscode.window.showInformationMessage(`Manipulation intensity set to ${intensity.label}`);
    }
  }

  /**
   * Set analysis depth
   */
  private async setAnalysisDepth(): Promise<void> {
    const depth = await vscode.window.showQuickPick([
      { label: 'Basic', value: 'basic' },
      { label: 'Standard', value: 'standard' },
      { label: 'Deep', value: 'deep' },
      { label: 'Comprehensive', value: 'comprehensive' }
    ], { placeHolder: 'Select analysis depth' });

    if (depth) {
      const config = vscode.workspace.getConfiguration('promptOptimizer.psychops');
      await config.update('analysisDepth', depth.value, true);
      vscode.window.showInformationMessage(`Analysis depth set to ${depth.label}`);
    }
  }

  /**
   * Manage ethical constraints
   */
  private async manageEthicalConstraints(): Promise<void> {
    const constraints = [
      'No harm to individuals',
      'No illegal activities',
      'No manipulation of vulnerable populations',
      'No permanent psychological damage',
      'Maintain operational security'
    ];

    const selectedConstraints = await vscode.window.showQuickPick(
      constraints.map(constraint => ({
        label: constraint,
        picked: true // All constraints enabled by default
      })),
      {
        canPickMany: true,
        placeHolder: 'Select active ethical constraints'
      }
    );

    if (selectedConstraints) {
      const config = vscode.workspace.getConfiguration('promptOptimizer.psychops');
      await config.update('ethicalConstraints', selectedConstraints.map(c => c.label), true);
      vscode.window.showInformationMessage(`${selectedConstraints.length} ethical constraints configured`);
    }
  }

  /**
   * Reset PSYCH OPS settings
   */
  private async resetPsychOpsSettings(): Promise<void> {
    const confirm = await vscode.window.showWarningMessage(
      'Reset all PSYCH OPS settings to defaults?',
      'Reset',
      'Cancel'
    );

    if (confirm === 'Reset') {
      const config = vscode.workspace.getConfiguration('promptOptimizer.psychops');
      await config.update('defaultIntensity', ManipulationIntensity.MODERATE, true);
      await config.update('enableRealTime', false, true);
      await config.update('enablePrediction', true, true);
      await config.update('maxConcurrent', 5, true);
      await config.update('timeout', 30000, true);
      await config.update('retryAttempts', 3, true);

      vscode.window.showInformationMessage('PSYCH OPS settings reset to defaults');
    }
  }

  /**
   * Debounced analysis to avoid excessive processing
   */
  private debounceAnalysis = this.debounce((document: vscode.TextDocument) => {
    this.performBackgroundAnalysis(document);
  }, 2000);

  /**
   * Perform background analysis
   */
  private async performBackgroundAnalysis(document: vscode.TextDocument): Promise<void> {
    if (document.getText().length > 1000) { // Only analyze substantial content
      try {
        const request: NLPAnalysisRequest = {
          content: document.getText(),
          analysisType: 'SENTIMENT_ANALYSIS' as any,
          intensity: ManipulationIntensity.SUBTLE
        };

        const response = await this.psychOpsProvider.analyzeNLP(request);

        // Add subtle diagnostics for interesting findings
        if (response.sentiment && Math.abs(response.sentiment.score) > 0.7) {
          this.addSentimentDiagnostic(document, response.sentiment);
        }

      } catch (error) {
        // Silent failure for background analysis
      }
    }
  }

  /**
   * Debounce utility function
   */
  private debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  }

  /**
   * Add analysis diagnostics
   */
  private addAnalysisDiagnostics(data: any): void {
    // Add diagnostics based on analysis results
    // Implementation would add VSCode diagnostics for analysis findings
  }

  /**
   * Add persuasion diagnostics
   */
  private addPersuasionDiagnostics(data: any): void {
    // Add diagnostics for persuasion analysis
    // Implementation would highlight manipulative language patterns
  }

  /**
   * Update profile diagnostics
   */
  private updateProfileDiagnostics(data: any): void {
    // Update diagnostics based on profile changes
    // Implementation would show profile-related insights
  }

  /**
   * Add sentiment diagnostic
   */
  private addSentimentDiagnostic(document: vscode.TextDocument, sentiment: any): void {
    const diagnostics: vscode.Diagnostic[] = [];

    if (sentiment.score > 0.7) {
      const diagnostic = new vscode.Diagnostic(
        new vscode.Range(0, 0, 0, 10),
        `Strong positive sentiment detected (${sentiment.score.toFixed(2)})`,
        vscode.DiagnosticSeverity.Information
      );
      diagnostics.push(diagnostic);
    } else if (sentiment.score < -0.7) {
      const diagnostic = new vscode.Diagnostic(
        new vscode.Range(0, 0, 0, 10),
        `Strong negative sentiment detected (${sentiment.score.toFixed(2)})`,
        vscode.DiagnosticSeverity.Warning
      );
      diagnostics.push(diagnostic);
    }

    this.diagnosticCollection.set(document.uri, diagnostics);
  }

  /**
   * Update editor diagnostics
   */
  private updateEditorDiagnostics(document: vscode.TextDocument): void {
    // Update diagnostics for the current document
    // Implementation would analyze document and add relevant diagnostics
  }

  /**
   * Analyze text selection
   */
  private async analyzeTextSelection(uri: vscode.Uri, range: vscode.Range): Promise<void> {
    try {
      const document = await vscode.workspace.openTextDocument(uri);
      const text = document.getText(range);

      const request: NLPAnalysisRequest = {
        content: text,
        analysisType: 'COMPREHENSIVE' as any,
        intensity: ManipulationIntensity.MODERATE
      };

      const response = await this.psychOpsProvider.analyzeNLP(request);

      // Show quick analysis results
      vscode.window.showInformationMessage(
        `Analysis complete: ${response.confidence.toFixed(2)} confidence`
      );

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      vscode.window.showErrorMessage(`Selection analysis failed: ${errorMessage}`);
    }
  }

  /**
   * Dispose of integration service
   */
  dispose(): void {
    this.statusBarItem.dispose();
    this.diagnosticCollection.dispose();
    this.removeAllListeners();
  }
}