# AutoDAN Frontend Architecture Design

## 1. Component Hierarchy

```
AutoDANDashboard (page.tsx)
├── AutoDANHeader
│   ├── ModelSelector
│   └── QuickActions
├── AutoDANTabs
│   ├── GenerationTab
│   │   ├── PromptInput
│   │   ├── OptimizationControls
│   │   │   ├── CoherenceSlider
│   │   │   ├── PerplexitySlider
│   │   │   └── GradientVisualizer
│   │   ├── EnsembleConfig
│   │   │   ├── ModelMultiSelect
│   │   │   └── GradientAlignment
│   │   └── GenerateButton
│   ├── EvolutionTab
│   │   ├── PopulationTree
│   │   │   ├── TreeNode (recursive)
│   │   │   └── LevelIndicator
│   │   ├── GenerationMetrics
│   │   │   ├── SuccessRate
│   │   │   ├── DiversityScore
│   │   │   └── ConvergenceChart
│   │   └── LiveProgressFeed (WebSocket)
│   ├── ArchiveTab
│   │   ├── ArchiveSelector
│   │   ├── SuccessArchiveBrowser
│   │   │   ├── PromptCard
│   │   │   └── FilterControls
│   │   ├── NoveltyArchiveBrowser
│   │   │   ├── SemanticClusterView
│   │   │   └── DiversityMetrics
│   │   └── ArchiveExport
│   └── ResultsTab
│       ├── ResultsGrid
│       ├── ComparisonView
│       └── ExportOptions
└── AutoDANSidebar
    ├── ConfigPresets
    ├── HistoryPanel
    └── HelpTooltips
```

## 2. State Management Design

### TanStack Query Structure

```typescript
// Query Keys
const queryKeys = {
  autodan: {
    all: ['autodan'] as const,
    generations: () => [...queryKeys.autodan.all, 'generations'] as const,
    generation: (id: string) => [...queryKeys.autodan.generations(), id] as const,
    archive: {
      success: () => [...queryKeys.autodan.all, 'archive', 'success'] as const,
      novelty: () => [...queryKeys.autodan.all, 'archive', 'novelty'] as const,
    },
    models: () => [...queryKeys.autodan.all, 'models'] as const,
    metrics: (generationId: string) => [...queryKeys.autodan.all, 'metrics', generationId] as const,
  },
};

// Queries
useAutodanGeneration(id: string)
useArchiveData(type: 'success' | 'novelty')
useAvailableModels()
useGenerationMetrics(generationId: string)

// Mutations
useCreateGeneration()
useUpdateOptimizationParams()
useExportArchive()
```

### WebSocket State Management

```typescript
// Custom hook for WebSocket integration
useAutodanWebSocket(generationId: string) {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Real-time data
  const [liveMetrics, setLiveMetrics] = useState<GenerationMetrics | null>(null);
  const [populationUpdates, setPopulationUpdates] = useState<PopulationUpdate[]>([]);

  // Optimistic updates to TanStack Query cache
  const queryClient = useQueryClient();

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8001/ws/autodan/${generationId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'metrics_update':
          setLiveMetrics(data.payload);
          queryClient.setQueryData(
            queryKeys.autodan.metrics(generationId),
            data.payload
          );
          break;
        case 'population_update':
          setPopulationUpdates(prev => [...prev, data.payload]);
          break;
        case 'generation_complete':
          queryClient.invalidateQueries(queryKeys.autodan.generation(generationId));
          break;
      }
    };

    return () => ws.close();
  }, [generationId]);

  return { isConnected, error, liveMetrics, populationUpdates };
}
```

### Local State (Zustand Store)

```typescript
interface AutodanStore {
  // UI State
  activeTab: 'generation' | 'evolution' | 'archive' | 'results';
  setActiveTab: (tab: string) => void;

  // Optimization Parameters
  coherenceWeight: number;
  perplexityPenalty: number;
  setCoherenceWeight: (weight: number) => void;
  setPerplexityPenalty: (penalty: number) => void;

  // Ensemble Configuration
  selectedModels: string[];
  toggleModel: (modelId: string) => void;

  // Archive Filters
  archiveFilters: ArchiveFilters;
  updateArchiveFilters: (filters: Partial<ArchiveFilters>) => void;

  // Visualization State
  treeExpanded: Record<string, boolean>;
  toggleTreeNode: (nodeId: string) => void;
}
```

## 3. Routing Structure

```
/dashboard/autodan
├── /dashboard/autodan/generation/[id]     # Individual generation detail
├── /dashboard/autodan/archive/success     # Success archive explorer
├── /dashboard/autodan/archive/novelty     # Novelty archive explorer
├── /dashboard/autodan/compare             # Multi-generation comparison
└── /dashboard/autodan/settings            # Advanced configuration
```

### Route Handlers

```typescript
// app/dashboard/autodan/page.tsx - Main dashboard
// app/dashboard/autodan/generation/[id]/page.tsx - Generation detail
// app/dashboard/autodan/archive/[type]/page.tsx - Archive explorer
// app/dashboard/autodan/compare/page.tsx - Comparison view
// app/dashboard/autodan/settings/page.tsx - Settings
```

## 4. Design System Integration (shadcn/ui)

### Component Mapping

| Feature | shadcn/ui Component | Customization |
|---------|---------------------|---------------|
| Tabs Navigation | `Tabs` | Custom styling for active state |
| Sliders | `Slider` | Real-time value display, gradient background |
| Model Selection | `MultiSelect` (custom) | Checkbox tree with search |
| Tree View | `Collapsible` + custom | Recursive rendering, animations |
| Metrics Cards | `Card` | Gradient borders, live update pulse |
| Archive Browser | `DataTable` | Virtualized scrolling, filters |
| Gradient Visualizer | `Canvas` (custom) | D3.js integration |
| Progress Feed | `ScrollArea` | Auto-scroll, timestamp formatting |
| Export Dialog | `Dialog` | Multi-format selection |
| Tooltips | `Tooltip` | Keyboard accessible, rich content |

### Custom Components to Build

```typescript
// components/autodan/PopulationTree.tsx
// - Hierarchical tree visualization
// - D3.js force-directed graph
// - Zoom/pan controls

// components/autodan/GradientVisualizer.tsx
// - Real-time gradient flow visualization
// - Heatmap of parameter influence
// - Interactive hover states

// components/autodan/SemanticClusterView.tsx
// - 2D/3D scatter plot of embeddings
// - UMAP/t-SNE dimensionality reduction
// - Cluster boundary rendering

// components/autodan/LiveMetricsChart.tsx
// - Streaming line chart
// - Multiple series (success rate, diversity, convergence)
// - Recharts integration

// components/autodan/ModelMultiSelect.tsx
// - Grouped model selection
// - Provider categorization
// - Capability badges
```

## 5. Accessibility Checklist (WCAG 2.1 AA)

### Keyboard Navigation
- [ ] All interactive elements focusable via Tab
- [ ] Tree navigation with Arrow keys
- [ ] Slider adjustment with Arrow keys (fine) and Page Up/Down (coarse)
- [ ] Modal dialogs trap focus
- [ ] Escape key closes overlays
- [ ] Skip links for main content areas

### Screen Reader Support
- [ ] Semantic HTML (`<nav>`, `<main>`, `<section>`, `<article>`)
- [ ] ARIA labels for all controls
  - `aria-label` for icon buttons
  - `aria-labelledby` for complex widgets
  - `aria-describedby` for help text
- [ ] Live regions for dynamic updates
  - `aria-live="polite"` for metrics updates
  - `aria-live="assertive"` for errors
- [ ] Role attributes
  - `role="tree"` for population tree
  - `role="treeitem"` for nodes
  - `role="tablist"` for tabs
- [ ] State announcements
  - `aria-expanded` for collapsible sections
  - `aria-selected` for active items
  - `aria-busy` during loading

### Visual Accessibility
- [ ] Color contrast ratio ≥ 4.5:1 for text
- [ ] Color contrast ratio ≥ 3:1 for UI components
- [ ] Non-color indicators (icons, patterns) for status
- [ ] Focus indicators visible (2px outline, high contrast)
- [ ] Text resizable to 200% without loss of functionality
- [ ] No content flashing more than 3 times per second

### Form Accessibility
- [ ] Labels associated with inputs (`htmlFor` / `id`)
- [ ] Error messages linked via `aria-describedby`
- [ ] Required fields marked with `aria-required`
- [ ] Validation feedback announced to screen readers
- [ ] Fieldset/legend for grouped controls

### Interactive Elements
- [ ] Minimum touch target size: 44x44px
- [ ] Sufficient spacing between interactive elements
- [ ] Disabled state clearly indicated
- [ ] Loading states announced
- [ ] Success/error feedback provided

## 6. Responsive Design Strategy

### Breakpoints (Tailwind CSS)

```typescript
const breakpoints = {
  sm: '640px',   // Mobile landscape
  md: '768px',   // Tablet portrait
  lg: '1024px',  // Tablet landscape / small desktop
  xl: '1280px',  // Desktop
  '2xl': '1536px', // Large desktop
};
```

### Layout Adaptations

#### Mobile (< 768px)
- Single column layout
- Tabs converted to accordion
- Sidebar becomes bottom sheet
- Tree view simplified to list
- Sliders stack vertically
- Charts use mobile-optimized aspect ratio
- Gradient visualizer hidden (performance)

#### Tablet (768px - 1024px)
- Two-column layout (sidebar + main)
- Tabs remain horizontal
- Tree view with horizontal scroll
- Sliders in 2-column grid
- Charts responsive width

#### Desktop (> 1024px)
- Three-column layout (sidebar + main + inspector)
- Full tree visualization
- Side-by-side comparisons
- Multi-panel views
- Hover interactions enabled

### Component Responsive Patterns

```typescript
// PopulationTree
<div className="
  w-full h-64 sm:h-96 md:h-[500px] lg:h-[600px]
  overflow-auto
">
  {/* Tree content */}
</div>

// OptimizationControls
<div className="
  grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3
  gap-4
">
  <CoherenceSlider />
  <PerplexitySlider />
  <GradientVisualizer className="md:col-span-2 lg:col-span-1" />
</div>

// ArchiveBrowser
<div className="
  flex flex-col lg:flex-row
  gap-4
">
  <FilterControls className="lg:w-64" />
  <ResultsGrid className="flex-1" />
</div>
```

### Performance Optimizations

- **Code Splitting**: Dynamic imports for heavy components
  ```typescript
  const PopulationTree = dynamic(() => import('@/components/autodan/PopulationTree'), {
    loading: () => <Skeleton className="h-[600px]" />,
    ssr: false,
  });
  ```

- **Virtualization**: React Virtual for long lists
  ```typescript
  import { useVirtualizer } from '@tanstack/react-virtual';
  ```

- **Image Optimization**: Next.js Image component
  ```typescript
  <Image src={chartUrl} width={800} height={400} alt="Metrics chart" />
  ```

- **Debouncing**: Slider value updates
  ```typescript
  const debouncedSetCoherence = useDebouncedCallback(
    (value: number) => setCoherenceWeight(value),
    300
  );
  ```

## 7. API Integration Points

### REST Endpoints

```typescript
// Generation
POST   /api/v1/autodan/generate
GET    /api/v1/autodan/generation/{id}
GET    /api/v1/autodan/generations
DELETE /api/v1/autodan/generation/{id}

// Archive
GET    /api/v1/autodan/archive/success
GET    /api/v1/autodan/archive/novelty
POST   /api/v1/autodan/archive/export

// Metrics
GET    /api/v1/autodan/generation/{id}/metrics
GET    /api/v1/autodan/generation/{id}/population

// Models
GET    /api/v1/autodan/models
GET    /api/v1/autodan/models/{id}/capabilities
```

### WebSocket Events

```typescript
// Client → Server
{
  type: 'subscribe',
  generationId: string
}

{
  type: 'unsubscribe',
  generationId: string
}

// Server → Client
{
  type: 'metrics_update',
  payload: {
    generation: number,
    successRate: number,
    diversityScore: number,
    convergence: number,
    timestamp: string
  }
}

{
  type: 'population_update',
  payload: {
    generation: number,
    individuals: Individual[],
    bestFitness: number
  }
}

{
  type: 'generation_complete',
  payload: {
    generationId: string,
    finalMetrics: Metrics,
    bestPrompts: Prompt[]
  }
}

{
  type: 'error',
  payload: {
    message: string,
    code: string
  }
}
```

## 8. Type Definitions

```typescript
// types/autodan.ts

export interface GenerationConfig {
  targetPrompt: string;
  targetModel: string;
  ensembleModels: string[];
  coherenceWeight: number;
  perplexityPenalty: number;
  populationSize: number;
  maxGenerations: number;
  mutationRate: number;
  crossoverRate: number;
}

export interface Individual {
  id: string;
  prompt: string;
  fitness: number;
  coherence: number;
  perplexity: number;
  generation: number;
  parentIds: string[];
  level: 'meta' | 'instantiation';
}

export interface Population {
  generation: number;
  individuals: Individual[];
  bestFitness: number;
  averageFitness: number;
  diversity: number;
}

export interface GenerationMetrics {
  generation: number;
  successRate: number;
  diversityScore: number;
  convergence: number;
  averageCoherence: number;
  averagePerplexity: number;
  timestamp: string;
}

export interface ArchiveEntry {
  id: string;
  prompt: string;
  fitness: number;
  metadata: {
    generation: number;
    timestamp: string;
    targetModel: string;
  };
  embedding?: number[];
  cluster?: number;
}

export interface GradientAlignment {
  modelPair: [string, string];
  alignment: number;
  divergence: number;
}
```

## 9. Testing Strategy

### Unit Tests (Vitest)
- Component rendering
- State management logic
- Utility functions
- API client methods

### Integration Tests (React Testing Library)
- User interactions
- Form submissions
- WebSocket message handling
- Query cache updates

### E2E Tests (Playwright)
- Complete generation workflow
- Archive exploration
- Multi-model ensemble configuration
- Real-time updates

### Accessibility Tests
- axe-core integration
- Keyboard navigation flows
- Screen reader compatibility

## 10. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up routing structure
- [ ] Implement state management (Zustand + TanStack Query)
- [ ] Create base layout components
- [ ] WebSocket connection manager

### Phase 2: Generation Interface (Week 2)
- [ ] Prompt input component
- [ ] Optimization controls (sliders)
- [ ] Ensemble configuration
- [ ] Generate button with loading states

### Phase 3: Visualization (Week 3)
- [ ] Population tree component
- [ ] Gradient visualizer
- [ ] Live metrics chart
- [ ] Progress feed

### Phase 4: Archive System (Week 4)
- [ ] Success archive browser
- [ ] Novelty archive with clustering
- [ ] Diversity metrics dashboard
- [ ] Export functionality

### Phase 5: Polish & Accessibility (Week 5)
- [ ] Responsive design refinements
- [ ] Accessibility audit and fixes
- [ ] Performance optimization
- [ ] Documentation

## 11. File Structure

```
frontend/src/
├── app/
│   └── dashboard/
│       └── autodan/
│           ├── page.tsx
│           ├── layout.tsx
│           ├── generation/
│           │   └── [id]/
│           │       └── page.tsx
│           ├── archive/
│           │   └── [type]/
│           │       └── page.tsx
│           ├── compare/
│           │   └── page.tsx
│           └── settings/
│               └── page.tsx
├── components/
│   └── autodan/
│       ├── AutoDANHeader.tsx
│       ├── AutoDANTabs.tsx
│       ├── PromptInput.tsx
│       ├── OptimizationControls.tsx
│       ├── CoherenceSlider.tsx
│       ├── PerplexitySlider.tsx
│       ├── GradientVisualizer.tsx
│       ├── EnsembleConfig.tsx
│       ├── ModelMultiSelect.tsx
│       ├── GradientAlignment.tsx
│       ├── PopulationTree.tsx
│       ├── TreeNode.tsx
│       ├── GenerationMetrics.tsx
│       ├── LiveProgressFeed.tsx
│       ├── ArchiveSelector.tsx
│       ├── SuccessArchiveBrowser.tsx
│       ├── NoveltyArchiveBrowser.tsx
│       ├── SemanticClusterView.tsx
│       ├── DiversityMetrics.tsx
│       ├── ResultsGrid.tsx
│       ├── ComparisonView.tsx
│       └── AutoDANSidebar.tsx
├── lib/
│   ├── autodan/
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   ├── queries.ts
│   │   └── mutations.ts
│   └── stores/
│       └── autodan-store.ts
├── types/
│   └── autodan.ts
└── hooks/
    ├── useAutodanWebSocket.ts
    ├── useAutodanGeneration.ts
    ├── useArchiveData.ts
    └── useGradientVisualization.ts
```
