/**
 * Interactive Help & Documentation Hub Interface
 *
 * Phase 2 feature for competitive differentiation:
 * - In-app searchable documentation
 * - Contextual help and best practices
 * - Ethical guidelines and legal considerations
 * - Video tutorials and getting started guides
 */

"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  BookOpen,
  Search,
  Video,
  HelpCircle,
  Rocket,
  Shield,
  Zap,
  Users,
  FileText,
  Clock,
  Eye,
  Star,
  Filter,
  ArrowRight,
  PlayCircle,
  CheckCircle,
  AlertTriangle,
  ExternalLink,
  ChevronRight,
  RefreshCw,
  Lightbulb
} from 'lucide-react';
import { toast } from 'sonner';
import { usePathname } from 'next/navigation';

// Import services
import {
  documentationService,
  DocumentationItem,
  DocumentationListResponse,
  SearchResult,
  VideoTutorial,
  QuickStartGuide,
  DocumentationType,
  DocumentationCategory,
  SearchRequest,
  DocumentationListParams
} from '@/lib/api/services/documentation';

export default function DocumentationHubPage() {
  // Data state
  const [documentation, setDocumentation] = useState<DocumentationListResponse | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [videoTutorials, setVideoTutorials] = useState<VideoTutorial[]>([]);
  const [quickStart, setQuickStart] = useState<QuickStartGuide | null>(null);
  const [selectedItem, setSelectedItem] = useState<DocumentationItem | null>(null);

  // Filter and search state
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<DocumentationListParams>({
    page: 1,
    page_size: 20
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [showItemDialog, setShowItemDialog] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const pathname = usePathname();

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (filters.category || filters.type || filters.difficulty) {
      loadDocumentation();
    }
  }, [filters]);

  const loadInitialData = useCallback(async () => {
    try {
      setLoading(true);

      const [docsData, videosData, quickStartData] = await Promise.all([
        documentationService.listDocumentation({ page: 1, page_size: 20 }),
        documentationService.listVideoTutorials(),
        documentationService.getQuickStartGuide()
      ]);

      setDocumentation(docsData);
      setVideoTutorials(videosData);
      setQuickStart(quickStartData);
    } catch (error) {
      // Errors already handled in services
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDocumentation = useCallback(async () => {
    try {
      const data = await documentationService.listDocumentation(filters);
      setDocumentation(data);
    } catch (error) {
      // Error already handled in service
    }
  }, [filters]);

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    try {
      setSearching(true);

      const searchRequest: SearchRequest = {
        query: searchQuery,
        category: filters.category,
        type: filters.type,
        difficulty_level: filters.difficulty
      };

      const results = await documentationService.searchDocumentation(searchRequest);
      setSearchResults(results);
      setActiveTab('search');
    } catch (error) {
      // Error already handled in service
    } finally {
      setSearching(false);
    }
  }, [searchQuery, filters]);

  const handleViewItem = useCallback(async (itemId: string) => {
    try {
      const item = await documentationService.getDocumentationItem(itemId);
      setSelectedItem(item);
      setShowItemDialog(true);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleFilterChange = useCallback((key: keyof DocumentationListParams, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value, page: 1 }));
  }, []);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  }, [handleSearch]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Documentation...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load the documentation hub.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Documentation & Help Hub
        </h1>
        <p className="text-muted-foreground text-lg">
          Comprehensive guides, tutorials, and contextual help for mastering Chimera&apos;s AI security testing platform.
        </p>
      </div>

      {/* Search Bar */}
      <div className="flex gap-4 px-4">
        <div className="flex-1 flex gap-2">
          <Input
            placeholder="Search documentation, guides, and tutorials..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1"
          />
          <Button onClick={handleSearch} disabled={searching}>
            {searching ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="search">Search</TabsTrigger>
          <TabsTrigger value="categories">Browse</TabsTrigger>
          <TabsTrigger value="videos">Videos</TabsTrigger>
          <TabsTrigger value="quick-start">Quick Start</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <OverviewView
            documentation={documentation}
            quickStart={quickStart}
            onViewItem={handleViewItem}
          />
        </TabsContent>

        <TabsContent value="search" className="space-y-6">
          <SearchResultsView
            results={searchResults}
            query={searchQuery}
            onViewItem={handleViewItem}
          />
        </TabsContent>

        <TabsContent value="categories" className="space-y-6">
          <CategoryBrowserView
            documentation={documentation}
            filters={filters}
            onFilterChange={handleFilterChange}
            onViewItem={handleViewItem}
            onRefresh={loadDocumentation}
          />
        </TabsContent>

        <TabsContent value="videos" className="space-y-6">
          <VideoTutorialsView tutorials={videoTutorials} />
        </TabsContent>

        <TabsContent value="quick-start" className="space-y-6">
          <QuickStartView quickStart={quickStart} />
        </TabsContent>
      </Tabs>

      {/* Documentation Item Dialog */}
      <Dialog open={showItemDialog} onOpenChange={setShowItemDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedItem && (
                <>
                  <BookOpen className="h-5 w-5" />
                  {selectedItem.title}
                </>
              )}
            </DialogTitle>
            <DialogDescription>
              {selectedItem && (
                <div className="flex items-center gap-4 text-sm">
                  <Badge variant="outline" className={`bg-${documentationService.getTypeBadgeColor(selectedItem.type)}-50`}>
                    {documentationService.getTypeDisplayName(selectedItem.type)}
                  </Badge>
                  <span className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {documentationService.formatReadingTime(selectedItem.estimated_read_time)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Eye className="h-4 w-4" />
                    {selectedItem.views} views
                  </span>
                </div>
              )}
            </DialogDescription>
          </DialogHeader>

          {selectedItem && (
            <ScrollArea className="h-[60vh] pr-4">
              <DocumentationContent item={selectedItem} />
            </ScrollArea>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Overview View Component
function OverviewView({
  documentation,
  quickStart,
  onViewItem
}: {
  documentation: DocumentationListResponse | null;
  quickStart: QuickStartGuide | null;
  onViewItem: (itemId: string) => void;
}) {
  if (!documentation || !quickStart) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="animate-pulse space-y-3">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded w-full"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Quick Access */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => onViewItem('getting-started-overview')}>
          <CardContent className="p-6 text-center">
            <Rocket className="h-8 w-8 mx-auto mb-3 text-blue-600" />
            <h3 className="font-semibold mb-2">Getting Started</h3>
            <p className="text-sm text-muted-foreground">Setup guide and first steps</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => onViewItem('ethical-guidelines')}>
          <CardContent className="p-6 text-center">
            <Shield className="h-8 w-8 mx-auto mb-3 text-green-600" />
            <h3 className="font-semibold mb-2">Ethical Guidelines</h3>
            <p className="text-sm text-muted-foreground">Responsible testing practices</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => onViewItem('attack-techniques-overview')}>
          <CardContent className="p-6 text-center">
            <Zap className="h-8 w-8 mx-auto mb-3 text-orange-600" />
            <h3 className="font-semibold mb-2">Attack Techniques</h3>
            <p className="text-sm text-muted-foreground">Comprehensive technique guide</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <HelpCircle className="h-8 w-8 mx-auto mb-3 text-purple-600" />
            <h3 className="font-semibold mb-2">FAQ</h3>
            <p className="text-sm text-muted-foreground">Common questions answered</p>
          </CardContent>
        </Card>
      </div>

      {/* Categories */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Browse by Category</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {documentation.categories.map((category) => (
            <Card key={category.id} className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{documentationService.getCategoryIcon(category.id as DocumentationCategory)}</div>
                  <div>
                    <h3 className="font-semibold">{category.name}</h3>
                    <p className="text-sm text-muted-foreground">{category.item_count} articles</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Recent & Popular */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h2 className="text-xl font-bold mb-4">Popular Articles</h2>
          <div className="space-y-3">
            {documentation.items.slice(0, 5).map((item) => (
              <Card key={item.id} className="cursor-pointer hover:shadow-sm transition-shadow" onClick={() => onViewItem(item.id)}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-medium mb-1">{item.title}</h4>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Badge variant="outline" className="text-xs">
                          {documentationService.getTypeDisplayName(item.type)}
                        </Badge>
                        <span className="flex items-center gap-1">
                          <Eye className="h-3 w-3" />
                          {item.views}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {documentationService.formatReadingTime(item.estimated_read_time)}
                        </span>
                      </div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <div>
          <h2 className="text-xl font-bold mb-4">Your Progress</h2>
          <Card>
            <CardContent className="p-4">
              <div className="space-y-4">
                {quickStart.next_steps.slice(0, 3).map((step, index) => (
                  <div key={index} className="flex items-center gap-3">
                    {step.priority === 'high' ? (
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                    ) : (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    )}
                    <div className="flex-1">
                      <h4 className="font-medium">{step.title}</h4>
                      <p className="text-sm text-muted-foreground">{step.description}</p>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {step.estimated_time}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

// Search Results View Component
function SearchResultsView({
  results,
  query,
  onViewItem
}: {
  results: SearchResult[];
  query: string;
  onViewItem: (itemId: string) => void;
}) {
  if (!query) {
    return (
      <div className="text-center py-12">
        <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">Search Documentation</h3>
        <p className="text-muted-foreground">
          Enter a search query above to find relevant documentation and guides.
        </p>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Results Found</h3>
        <p className="text-muted-foreground">
          No documentation found for &quot;{query}&quot;. Try different keywords or browse by category.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Search Results</h2>
        <p className="text-muted-foreground">
          Found {results.length} result{results.length > 1 ? 's' : ''} for &quot;{query}&quot;
        </p>
      </div>

      <div className="space-y-4">
        {results.map((result, index) => (
          <Card key={result.item.id} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => onViewItem(result.item.id)}>
            <CardContent className="p-6">
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-semibold text-lg">{result.item.title}</h3>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">
                    {documentationService.getTypeDisplayName(result.item.type)}
                  </Badge>
                  <Badge variant="secondary">
                    Score: {result.relevance_score.toFixed(1)}
                  </Badge>
                </div>
              </div>

              {result.highlight_text && (
                <p className="text-muted-foreground mb-3 line-clamp-2">
                  {result.highlight_text}
                </p>
              )}

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {documentationService.formatReadingTime(result.item.estimated_read_time)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Eye className="h-4 w-4" />
                    {result.item.views} views
                  </span>
                  {result.matching_sections.length > 0 && (
                    <span className="flex items-center gap-1">
                      <FileText className="h-4 w-4" />
                      {result.matching_sections.length} matching section{result.matching_sections.length > 1 ? 's' : ''}
                    </span>
                  )}
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

// Category Browser View Component
function CategoryBrowserView({
  documentation,
  filters,
  onFilterChange,
  onViewItem,
  onRefresh
}: {
  documentation: DocumentationListResponse | null;
  filters: DocumentationListParams;
  onFilterChange: (key: keyof DocumentationListParams, value: any) => void;
  onViewItem: (itemId: string) => void;
  onRefresh: () => void;
}) {
  if (!documentation) return null;

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="flex flex-wrap gap-4 items-center">
        <Select
          value={filters.category || 'all'}
          onValueChange={(value) => onFilterChange('category', value === 'all' ? undefined : value)}
        >
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {documentation.categories.map((category) => (
              <SelectItem key={category.id} value={category.id}>
                {category.name} ({category.item_count})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={filters.type || 'all'}
          onValueChange={(value) => onFilterChange('type', value === 'all' ? undefined : value)}
        >
          <SelectTrigger className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="tutorial">Tutorial</SelectItem>
            <SelectItem value="guide">Guide</SelectItem>
            <SelectItem value="reference">Reference</SelectItem>
            <SelectItem value="faq">FAQ</SelectItem>
            <SelectItem value="ethical_guidelines">Ethics</SelectItem>
          </SelectContent>
        </Select>

        <Select
          value={filters.difficulty || 'all'}
          onValueChange={(value) => onFilterChange('difficulty', value === 'all' ? undefined : value)}
        >
          <SelectTrigger className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Levels</SelectItem>
            <SelectItem value="beginner">Beginner</SelectItem>
            <SelectItem value="intermediate">Intermediate</SelectItem>
            <SelectItem value="advanced">Advanced</SelectItem>
          </SelectContent>
        </Select>

        <Button onClick={onRefresh} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Documentation Grid */}
      {documentation.items.length === 0 ? (
        <Card>
          <CardContent className="text-center py-12">
            <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Documentation Found</h3>
            <p className="text-muted-foreground">
              No items match your current filters. Try adjusting the filters or browse all categories.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {documentation.items.map((item) => (
            <Card key={item.id} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => onViewItem(item.id)}>
              <CardContent className="p-6">
                <div className="flex items-start justify-between mb-3">
                  <Badge variant="outline" className={`bg-${documentationService.getTypeBadgeColor(item.type)}-50`}>
                    {documentationService.getTypeDisplayName(item.type)}
                  </Badge>
                  <Badge variant="secondary" className={`bg-${documentationService.getDifficultyColor(item.difficulty_level)}-50`}>
                    {item.difficulty_level}
                  </Badge>
                </div>

                <h3 className="font-semibold mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground mb-4 line-clamp-3">
                  {item.content.substring(0, 150)}...
                </p>

                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {documentationService.formatReadingTime(item.estimated_read_time)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Eye className="h-3 w-3" />
                    {item.views}
                  </span>
                </div>

                {item.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-3">
                    {item.tags.slice(0, 3).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                    {item.tags.length > 3 && (
                      <Badge variant="secondary" className="text-xs">
                        +{item.tags.length - 3}
                      </Badge>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Pagination */}
      {documentation.total > (documentation.page_size || 20) && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Showing {(((documentation.page || 1) - 1) * (documentation.page_size || 20)) + 1} to {Math.min((documentation.page || 1) * (documentation.page_size || 20), documentation.total)} of {documentation.total} items
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={!documentation.has_prev}
              onClick={() => onFilterChange('page', (documentation.page || 1) - 1)}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!documentation.has_next}
              onClick={() => onFilterChange('page', (documentation.page || 1) + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// Video Tutorials View Component
function VideoTutorialsView({ tutorials }: { tutorials: VideoTutorial[] }) {
  if (tutorials.length === 0) {
    return (
      <Card>
        <CardContent className="text-center py-12">
          <Video className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Video Tutorials Available</h3>
          <p className="text-muted-foreground">
            Video tutorials will be available soon to help you get started with Chimera.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {tutorials.map((tutorial) => (
        <Card key={tutorial.id} className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-0">
            <div className="relative">
              {tutorial.thumbnail_url ? (
                <img
                  src={tutorial.thumbnail_url}
                  alt={tutorial.title}
                  className="w-full h-48 object-cover rounded-t-lg"
                />
              ) : (
                <div className="w-full h-48 bg-gradient-to-br from-blue-500 to-purple-600 rounded-t-lg flex items-center justify-center">
                  <PlayCircle className="h-16 w-16 text-white" />
                </div>
              )}
              <div className="absolute bottom-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                {documentationService.formatVideoDuration(tutorial.duration)}
              </div>
            </div>
            <div className="p-4">
              <h3 className="font-semibold mb-2">{tutorial.title}</h3>
              <p className="text-sm text-muted-foreground mb-3">{tutorial.description}</p>
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Eye className="h-3 w-3" />
                  {tutorial.views} views
                </span>
                <span>{new Date(tutorial.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// Quick Start View Component
function QuickStartView({ quickStart }: { quickStart: QuickStartGuide | null }) {
  if (!quickStart) return null;

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            {quickStart.welcome_message}
          </CardTitle>
          <CardDescription>
            Follow these steps to get the most out of Chimera&apos;s AI security testing platform.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {quickStart.next_steps.map((step, index) => (
              <div key={index} className="flex items-start gap-4 p-4 border rounded-lg">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                  step.priority === 'high' ? 'bg-red-100 text-red-700' :
                  step.priority === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-green-100 text-green-700'
                }`}>
                  {index + 1}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold">{step.title}</h3>
                    <Badge variant="outline" className="text-xs">
                      {step.estimated_time}
                    </Badge>
                    <Badge variant={step.priority === 'high' ? 'destructive' : step.priority === 'medium' ? 'default' : 'secondary'} className="text-xs">
                      {step.priority}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">{step.description}</p>
                  <Button size="sm" variant="outline" asChild>
                    <a href={step.target}>
                      <ArrowRight className="h-4 w-4 mr-1" />
                      {step.action === 'navigate' ? 'Go to' : 'Learn more'}
                    </a>
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div>
        <h2 className="text-xl font-bold mb-4">Featured Content</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {quickStart.featured_content.filter(Boolean).map((item) => (
            <Card key={item.id} className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className={`bg-${documentationService.getTypeBadgeColor(item.type)}-50`}>
                    {documentationService.getTypeDisplayName(item.type)}
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    {documentationService.formatReadingTime(item.estimated_read_time)}
                  </Badge>
                </div>
                <h3 className="font-semibold mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {item.content.substring(0, 100)}...
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

// Documentation Content Component
function DocumentationContent({ item }: { item: DocumentationItem }) {
  return (
    <div className="space-y-6">
      {/* Article Info */}
      <div className="flex flex-wrap items-center gap-4 pb-4 border-b">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className={`bg-${documentationService.getTypeBadgeColor(item.type)}-50`}>
            {documentationService.getTypeDisplayName(item.type)}
          </Badge>
          <Badge variant="secondary" className={`bg-${documentationService.getDifficultyColor(item.difficulty_level)}-50`}>
            {item.difficulty_level}
          </Badge>
        </div>
        <div className="text-sm text-muted-foreground flex items-center gap-4">
          <span className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            {documentationService.formatReadingTime(item.estimated_read_time)}
          </span>
          <span className="flex items-center gap-1">
            <Eye className="h-4 w-4" />
            {item.views} views
          </span>
          <span>Updated {new Date(item.last_updated).toLocaleDateString()}</span>
        </div>
      </div>

      {/* Table of Contents */}
      {item.sections.length > 0 && (
        <div className="bg-muted rounded-lg p-4">
          <h4 className="font-semibold mb-3">Table of Contents</h4>
          <ul className="space-y-1">
            {item.sections.map((section, index) => (
              <li key={index}>
                <a
                  href={`#${section.anchor}`}
                  className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
                >
                  {section.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Content */}
      <div
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{
          __html: documentationService.processMarkdownContent(item.content)
        }}
      />

      {/* Tags */}
      {item.tags.length > 0 && (
        <div>
          <h4 className="font-semibold mb-2">Tags</h4>
          <div className="flex flex-wrap gap-2">
            {item.tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Related Articles */}
      {item.related_articles.length > 0 && (
        <div>
          <h4 className="font-semibold mb-3">Related Articles</h4>
          <div className="space-y-2">
            {item.related_articles.map((articleId) => (
              <div key={articleId} className="flex items-center gap-2 text-sm">
                <ArrowRight className="h-4 w-4" />
                <span className="text-blue-600 hover:text-blue-800 cursor-pointer">
                  {articleId}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Video */}
      {item.video_url && (
        <div>
          <h4 className="font-semibold mb-3">Video Tutorial</h4>
          <div className="bg-gray-100 rounded-lg p-4 text-center">
            <PlayCircle className="h-12 w-12 mx-auto mb-2 text-blue-600" />
            <p className="text-sm text-muted-foreground mb-3">Watch the video tutorial for this topic</p>
            <Button size="sm" asChild>
              <a href={item.video_url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4 mr-1" />
                Watch Video
              </a>
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}