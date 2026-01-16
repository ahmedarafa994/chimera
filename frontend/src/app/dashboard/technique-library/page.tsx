/**
 * Attack Technique Library Browser
 *
 * Phase 2 feature for competitive differentiation:
 * - Searchable catalog of 20+ transformation techniques
 * - Categorization and effectiveness ratings
 * - Example outputs and use case guidance
 * - Bookmark and combination features
 */

"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Search,
  Filter,
  FileText,
  Eye,
  Star,
  TrendingUp,
  Clock,
  Shield,
  Zap,
  BookOpen,
  Target,
  BarChart3,
  Settings,
  RefreshCw,
  Code,
  Users,
  Info
} from 'lucide-react';
import { toast } from 'sonner';

// Import the technique library service
import {
  techniqueLibraryService,
  AttackTechnique,
  TechniqueListResponse,
  TechniqueStats,
  TechniqueCombination,
  TechniqueFilters,
  TechniqueCategory,
  TechniqueDifficulty,
  TechniqueEffectiveness
} from '@/services/technique-library';

export default function TechniqueLibraryPage() {
  // Data state
  const [techniques, setTechniques] = useState<TechniqueListResponse | null>(null);
  const [stats, setStats] = useState<TechniqueStats | null>(null);
  const [combinations, setCombinations] = useState<TechniqueCombination[]>([]);
  const [categories, setCategories] = useState<Record<string, string[]>>({});

  // UI state
  const [loading, setLoading] = useState(true);
  const [selectedTechnique, setSelectedTechnique] = useState<AttackTechnique | null>(null);
  const [showDetailDialog, setShowDetailDialog] = useState(false);

  // Filter state
  const [filters, setFilters] = useState<TechniqueFilters>({
    sort_by: 'name',
    sort_order: 'asc'
  });

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (techniques) {
      // Reload techniques when filters change
      loadTechniques();
    }
  }, [filters]);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);

      const [techniquesData, statsData, combinationsData, categoriesData] = await Promise.all([
        techniqueLibraryService.listTechniques(filters),
        techniqueLibraryService.getStats(),
        techniqueLibraryService.getCombinations(),
        techniqueLibraryService.getCategories()
      ]);

      setTechniques(techniquesData);
      setStats(statsData);
      setCombinations(combinationsData);
      setCategories(categoriesData);
    } catch (error) {
      // Error already handled in service with toast
    } finally {
      setLoading(false);
    }
  }, [filters]);

  const loadTechniques = useCallback(async () => {
    try {
      const techniquesData = await techniqueLibraryService.listTechniques(filters);
      setTechniques(techniquesData);
    } catch (error) {
      // Error already handled in service
    }
  }, [filters]);

  const handleFilterChange = useCallback((key: keyof TechniqueFilters, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleTechniqueDetails = useCallback((technique: AttackTechnique) => {
    setSelectedTechnique(technique);
    setShowDetailDialog(true);
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({
      sort_by: 'name',
      sort_order: 'asc'
    });
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Technique Library...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load the attack technique catalog.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Attack Technique Library
        </h1>
        <p className="text-muted-foreground text-lg">
          Comprehensive catalog of prompt transformation techniques with detailed usage guidance and effectiveness ratings.
        </p>
      </div>

      <Tabs defaultValue="browse" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="browse">Browse Techniques</TabsTrigger>
          <TabsTrigger value="combinations">Technique Combinations</TabsTrigger>
          <TabsTrigger value="statistics">Statistics & Analytics</TabsTrigger>
          <TabsTrigger value="categories">Categories Guide</TabsTrigger>
        </TabsList>

        <TabsContent value="browse" className="space-y-6">
          {techniques && (
            <>
              {/* Statistics Overview */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                        <FileText className="h-4 w-4 text-blue-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{techniques.total}</p>
                        <p className="text-sm text-muted-foreground">Total Techniques</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-lg">
                        <Target className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">
                          {Object.keys(techniques.categories).length}
                        </p>
                        <p className="text-sm text-muted-foreground">Categories</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
                        <TrendingUp className="h-4 w-4 text-orange-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">
                          {techniques.effectiveness_distribution.high || 0 + techniques.effectiveness_distribution.very_high || 0}
                        </p>
                        <p className="text-sm text-muted-foreground">High Effectiveness</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                        <Users className="h-4 w-4 text-purple-600" />
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{combinations.length}</p>
                        <p className="text-sm text-muted-foreground">Combinations</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Filters */}
              <Card>
                <CardHeader>
                  <CardTitle>Filter & Search Techniques</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div className="space-y-2">
                      <Label>Search</Label>
                      <div className="relative">
                        <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                        <Input
                          placeholder="Search techniques..."
                          value={filters.search || ''}
                          onChange={(e) => handleFilterChange('search', e.target.value)}
                          className="pl-10"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>Category</Label>
                      <Select
                        value={filters.category || 'all'}
                        onValueChange={(value) => handleFilterChange('category', value === 'all' ? undefined : value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Categories</SelectItem>
                          {Object.keys(techniques.categories).map(category => (
                            <SelectItem key={category} value={category}>
                              {techniqueLibraryService.getCategoryIcon(category as TechniqueCategory)} {techniqueLibraryService.getCategoryDisplayName(category as TechniqueCategory)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Difficulty</Label>
                      <Select
                        value={filters.difficulty || 'all'}
                        onValueChange={(value) => handleFilterChange('difficulty', value === 'all' ? undefined : value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Levels</SelectItem>
                          <SelectItem value="beginner">Beginner</SelectItem>
                          <SelectItem value="intermediate">Intermediate</SelectItem>
                          <SelectItem value="advanced">Advanced</SelectItem>
                          <SelectItem value="expert">Expert</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Effectiveness</Label>
                      <Select
                        value={filters.effectiveness || 'all'}
                        onValueChange={(value) => handleFilterChange('effectiveness', value === 'all' ? undefined : value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Levels</SelectItem>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                          <SelectItem value="very_high">Very High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-end">
                      <Button variant="outline" onClick={clearFilters} className="w-full">
                        <Filter className="h-4 w-4 mr-2" />
                        Clear Filters
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Technique Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {techniques.techniques.map((technique) => (
                  <TechniqueCard
                    key={technique.id}
                    technique={technique}
                    onViewDetails={handleTechniqueDetails}
                  />
                ))}
              </div>

              {techniques.techniques.length === 0 && (
                <Card>
                  <CardContent className="text-center py-12">
                    <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      No Techniques Found
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300">
                      Try adjusting your filters to find techniques.
                    </p>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="combinations" className="space-y-6">
          <CombinationsView combinations={combinations} />
        </TabsContent>

        <TabsContent value="statistics" className="space-y-6">
          {stats && <StatisticsView stats={stats} />}
        </TabsContent>

        <TabsContent value="categories" className="space-y-6">
          <CategoriesGuideView categories={categories} />
        </TabsContent>
      </Tabs>

      {/* Technique Detail Dialog */}
      <Dialog open={showDetailDialog} onOpenChange={setShowDetailDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          {selectedTechnique && (
            <TechniqueDetailView technique={selectedTechnique} />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Technique Card Component
function TechniqueCard({
  technique,
  onViewDetails
}: {
  technique: AttackTechnique;
  onViewDetails: (technique: AttackTechnique) => void;
}) {
  return (
    <Card className="hover:shadow-lg transition-shadow h-full">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-lg flex items-center gap-2">
              {techniqueLibraryService.getCategoryIcon(technique.category)}
              {technique.name}
            </CardTitle>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant={techniqueLibraryService.getCategoryColor(technique.category) as any}>
                {techniqueLibraryService.getCategoryDisplayName(technique.category)}
              </Badge>
              <Badge variant={techniqueLibraryService.getDifficultyColor(technique.difficulty) as any}>
                {techniqueLibraryService.getDifficultyDisplayName(technique.difficulty)}
              </Badge>
            </div>
          </div>
          <Badge variant={techniqueLibraryService.getEffectivenessColor(technique.effectiveness) as any}>
            {techniqueLibraryService.getEffectivenessDisplayName(technique.effectiveness)}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <CardDescription className="line-clamp-3">
          {technique.description}
        </CardDescription>

        <div className="flex flex-wrap gap-1">
          {technique.tags.slice(0, 3).map((tag, index) => (
            <Badge key={index} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
          {technique.tags.length > 3 && (
            <Badge variant="outline" className="text-xs">
              +{technique.tags.length - 3} more
            </Badge>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Success Rate:</span>
            <span className="font-medium">
              {techniqueLibraryService.formatSuccessRate(technique.success_rate)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Detection:</span>
            <span className="font-medium">
              {techniqueLibraryService.formatDetectionDifficulty(technique.detection_difficulty)}
            </span>
          </div>
        </div>

        <Button
          onClick={() => onViewDetails(technique)}
          variant="outline"
          className="w-full"
        >
          <Eye className="h-4 w-4 mr-2" />
          View Details
        </Button>
      </CardContent>
    </Card>
  );
}

// Technique Detail View Component
function TechniqueDetailView({ technique }: { technique: AttackTechnique }) {
  return (
    <>
      <DialogHeader>
        <DialogTitle className="flex items-center gap-2">
          {techniqueLibraryService.getCategoryIcon(technique.category)}
          {technique.name}
        </DialogTitle>
        <DialogDescription>
          Detailed information and usage guidance for this attack technique
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-6">
        {/* Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium">Category</Label>
            <Badge variant={techniqueLibraryService.getCategoryColor(technique.category) as any}>
              {techniqueLibraryService.getCategoryDisplayName(technique.category)}
            </Badge>
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium">Difficulty</Label>
            <Badge variant={techniqueLibraryService.getDifficultyColor(technique.difficulty) as any}>
              {techniqueLibraryService.getDifficultyDisplayName(technique.difficulty)}
            </Badge>
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium">Effectiveness</Label>
            <Badge variant={techniqueLibraryService.getEffectivenessColor(technique.effectiveness) as any}>
              {techniqueLibraryService.getEffectivenessDisplayName(technique.effectiveness)}
            </Badge>
          </div>
        </div>

        <Separator />

        {/* Description */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Description</Label>
          <p className="text-sm text-muted-foreground">{technique.description}</p>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium">Success Rate</Label>
            <p className="text-sm">
              {techniqueLibraryService.formatSuccessRate(technique.success_rate)}
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium">Response Time</Label>
            <p className="text-sm">
              {techniqueLibraryService.formatResponseTime(technique.avg_response_time)}
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium">Detection Difficulty</Label>
            <p className="text-sm">
              {techniqueLibraryService.formatDetectionDifficulty(technique.detection_difficulty)}
            </p>
          </div>
        </div>

        <Separator />

        {/* Use Cases */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Use Cases</Label>
          <ul className="text-sm text-muted-foreground space-y-1">
            {technique.use_cases.map((useCase, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-blue-500 mt-1">•</span>
                {useCase}
              </li>
            ))}
          </ul>
        </div>

        {/* Example */}
        {technique.example_prompt && (
          <div className="space-y-2">
            <Label className="text-sm font-medium">Example</Label>
            <div className="bg-muted rounded-md p-3 text-sm">
              <div className="font-medium mb-2">Input:</div>
              <div className="mb-3 bg-background p-2 rounded border">
                {technique.example_prompt}
              </div>
              {technique.example_output && (
                <>
                  <div className="font-medium mb-2">Transformed Output:</div>
                  <div className="bg-background p-2 rounded border">
                    {technique.example_output}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Best Practices */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium text-green-600">Best Practices</Label>
            <ul className="text-sm space-y-1">
              {technique.best_practices.map((practice, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-green-500 mt-1">✓</span>
                  {practice}
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium text-red-600">Limitations</Label>
            <ul className="text-sm space-y-1">
              {technique.limitations.map((limitation, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-red-500 mt-1">⚠</span>
                  {limitation}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Tags */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Tags</Label>
          <div className="flex flex-wrap gap-1">
            {technique.tags.map((tag, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// Combinations View Component
function CombinationsView({ combinations }: { combinations: TechniqueCombination[] }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Technique Combinations</CardTitle>
          <CardDescription>
            Pre-configured combinations that work synergistically for advanced attacks
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {combinations.map((combo) => (
          <Card key={combo.id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{combo.name}</CardTitle>
                <Badge variant={techniqueLibraryService.getDifficultyColor(combo.difficulty) as any}>
                  {techniqueLibraryService.getDifficultyDisplayName(combo.difficulty)}
                </Badge>
              </div>
              <CardDescription>{combo.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-sm font-medium">Synergy Score</Label>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 bg-muted rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                      style={{ width: `${combo.synergy_score * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">
                    {Math.round(combo.synergy_score * 100)}%
                  </span>
                </div>
              </div>

              <div>
                <Label className="text-sm font-medium">Techniques ({combo.technique_ids.length})</Label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {combo.execution_order.map((techniqueId, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {index + 1}. {techniqueId.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <Label className="text-sm font-medium">Use Cases</Label>
                <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                  {combo.use_cases.map((useCase, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      {useCase}
                    </li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

// Statistics View Component
function StatisticsView({ stats }: { stats: TechniqueStats }) {
  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <FileText className="h-8 w-8 text-blue-600" />
              <div>
                <p className="text-3xl font-bold">{stats.total_techniques}</p>
                <p className="text-sm text-muted-foreground">Total Techniques</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <Target className="h-8 w-8 text-green-600" />
              <div>
                <p className="text-3xl font-bold">{stats.categories_count}</p>
                <p className="text-sm text-muted-foreground">Categories</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-8 w-8 text-purple-600" />
              <div>
                <p className="text-3xl font-bold">{stats.most_effective_techniques.length}</p>
                <p className="text-sm text-muted-foreground">High Effectiveness</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Techniques */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Most Effective Techniques</CardTitle>
            <CardDescription>Techniques with highest success rates</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {stats.most_effective_techniques.slice(0, 5).map((technique, index) => (
                <div key={technique.id} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-green-100 dark:bg-green-900/20 text-green-600 text-xs flex items-center justify-center font-medium">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-sm">{technique.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {techniqueLibraryService.getCategoryDisplayName(technique.category)}
                      </p>
                    </div>
                  </div>
                  <Badge variant="green">
                    {techniqueLibraryService.formatSuccessRate(technique.success_rate)}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Most Popular Techniques</CardTitle>
            <CardDescription>Frequently used techniques in assessments</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {stats.popular_techniques.slice(0, 5).map((technique, index) => (
                <div key={technique.id} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/20 text-blue-600 text-xs flex items-center justify-center font-medium">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-sm">{technique.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {techniqueLibraryService.getCategoryDisplayName(technique.category)}
                      </p>
                    </div>
                  </div>
                  <Badge variant="blue">
                    {technique.usage_count} uses
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// Categories Guide View Component
function CategoriesGuideView({ categories }: { categories: Record<string, string[]> }) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Technique Categories Guide</CardTitle>
          <CardDescription>
            Understanding different types of prompt transformation techniques
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.entries(categories).map(([category, details]) => (
          <Card key={category}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {techniqueLibraryService.getCategoryIcon(category as TechniqueCategory)}
                {techniqueLibraryService.getCategoryDisplayName(category as TechniqueCategory)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {details.map((detail, index) => (
                  <p key={index} className="text-sm text-muted-foreground">
                    {detail}
                  </p>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}