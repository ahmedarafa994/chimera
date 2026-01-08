'use client';

import React, { Suspense, use, useDeferredValue, useTransition } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useEnhancedTransition, useConcurrentSearch, usePriorityRendering } from '@/lib/hooks/concurrent-hooks';
import { Loader2, Search } from 'lucide-react';

// Simulated data for demonstration
const generateMockData = async (query: string): Promise<Array<{ id: string; name: string; description: string }>> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 700));

  return Array.from({ length: 10 }, (_, i) => ({
    id: `item-${i}`,
    name: `${query} Result ${i + 1}`,
    description: `Description for ${query} item ${i + 1}`,
  }));
};

// Heavy computation simulation
const heavyComputation = (data: any[]) => {
  // Simulate CPU-intensive work
  let result = 0;
  for (let i = 0; i < data.length * 1000; i++) {
    result += Math.sqrt(i);
  }
  return data.map((item, index) => ({ ...item, computed: result + index }));
};

// Loading skeleton for search results
const SearchResultSkeleton = () => (
  <div className="space-y-2">
    {Array.from({ length: 5 }).map((_, i) => (
      <Card key={i}>
        <CardContent className="pt-4">
          <Skeleton className="h-4 w-3/4 mb-2" />
          <Skeleton className="h-3 w-1/2" />
        </CardContent>
      </Card>
    ))}
  </div>
);

// Concurrent search results component
function SearchResults({ results, isPending }: { results: any[]; isPending: boolean }) {
  // Use deferred value for expensive computations
  const deferredResults = useDeferredValue(results);
  const { value: processedResults, isPending: isComputePending } = usePriorityRendering(
    heavyComputation(deferredResults),
    { highPriority: false, deferMs: 200 }
  );

  const isStale = results !== deferredResults;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Search Results</h3>
        <div className="flex items-center space-x-2">
          {isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          {isStale && <Badge variant="secondary">Computing...</Badge>}
          {isComputePending && <Badge variant="outline">Processing</Badge>}
        </div>
      </div>

      <Suspense fallback={<SearchResultSkeleton />}>
        {processedResults.length > 0 ? (
          <div className="space-y-2">
            {processedResults.map((result) => (
              <Card key={result.id} className={isStale ? 'opacity-60' : ''}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium">{result.name}</h4>
                      <p className="text-sm text-muted-foreground">{result.description}</p>
                    </div>
                    <Badge variant="outline">
                      Score: {Math.round(result.computed % 1000)}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="pt-6">
              <p className="text-center text-muted-foreground">No results found</p>
            </CardContent>
          </Card>
        )}
      </Suspense>
    </div>
  );
}

// Main concurrent features demo component
export default function ConcurrentFeaturesDemo() {
  const [searchQuery, setSearchQuery] = React.useState('');

  // Enhanced transition for heavy operations
  const { isPending: isTransitionPending, startTransition } = useEnhancedTransition({
    timeoutMs: 10000,
    onStart: () => console.log('Heavy operation started'),
    onComplete: () => console.log('Heavy operation completed'),
    onError: (error) => console.error('Heavy operation failed:', error),
  });

  // Concurrent search with automatic debouncing
  const {
    query,
    results,
    error,
    isPending: isSearchPending,
    search,
    isStale,
  } = useConcurrentSearch(generateMockData, {
    debounceMs: 300,
    minQueryLength: 2,
  });

  // Handle search input
  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchQuery(value);
    search(value);
  };

  // Heavy operation simulation
  const performHeavyOperation = () => {
    startTransition(() => {
      // Simulate heavy computation
      const start = performance.now();
      let result = 0;
      for (let i = 0; i < 10000000; i++) {
        result += Math.sqrt(i);
      }
      const end = performance.now();
      console.log(`Heavy computation took ${end - start} milliseconds. Result: ${result}`);
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>React 19 Concurrent Features Demo</CardTitle>
          <CardDescription>
            Demonstrating useTransition, useDeferredValue, and concurrent rendering patterns
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Heavy Operation Demo */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Concurrent Transitions</h3>
            <div className="flex items-center space-x-4">
              <Button
                onClick={performHeavyOperation}
                disabled={isTransitionPending}
                className="flex items-center space-x-2"
              >
                {isTransitionPending && <Loader2 className="h-4 w-4 animate-spin" />}
                <span>
                  {isTransitionPending ? 'Computing...' : 'Start Heavy Operation'}
                </span>
              </Button>
              {isTransitionPending && (
                <Badge variant="secondary">
                  Non-blocking computation in progress
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              This heavy computation runs in a transition, keeping the UI responsive.
              You can still interact with the search below while it&apos;s running.
            </p>
          </div>

          {/* Concurrent Search Demo */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Concurrent Search with Deferred Values</h3>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search for items... (min 2 characters)"
                value={searchQuery}
                onChange={handleSearchInput}
                className="pl-10"
              />
            </div>

            {error && (
              <Card className="border-destructive">
                <CardContent className="pt-4">
                  <p className="text-destructive">Search error: {error.message}</p>
                </CardContent>
              </Card>
            )}

            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
              <span>Query: &quot;{query}&quot;</span>
              {isStale && <Badge variant="outline">Updating...</Badge>}
              {isSearchPending && <Loader2 className="h-3 w-3 animate-spin" />}
            </div>

            {query.length >= 2 && (
              <SearchResults
                results={results}
                isPending={isSearchPending}
              />
            )}
          </div>

          {/* Performance Insights */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Performance Insights</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {isTransitionPending ? '⏳' : '✅'}
                    </div>
                    <p className="text-xs text-muted-foreground">UI Responsive</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {isSearchPending ? '🔍' : '💤'}
                    </div>
                    <p className="text-xs text-muted-foreground">Search State</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-600">
                      {isStale ? '⚡' : '✨'}
                    </div>
                    <p className="text-xs text-muted-foreground">Deferred Updates</p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {results.length}
                    </div>
                    <p className="text-xs text-muted-foreground">Results</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}