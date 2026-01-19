"use client";

/**
 * Admin Dashboard Page for Project Chimera
 * 
 * Provides administrative functionality including:
 * - Feature flag management
 * - Tenant management
 * - Usage analytics
 * - System health monitoring
 */

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  useAdminAuth,
  useFeatureFlags,
  useTenants,
  useUsageAnalytics,
} from "@/hooks";
import {
  Shield,
  Users,
  Settings,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Plus,
  Trash2,
  RefreshCw,
  Lock,
} from "lucide-react";

// =============================================================================
// Auth Guard Component
// =============================================================================

// Create a context to share auth state across admin components
import { createContext, useContext } from "react";

interface AdminAuthContextType {
  authConfig: { apiKey: string } | null;
}

const AdminAuthContext = createContext<AdminAuthContextType>({ authConfig: null });

function useAdminAuthContext() {
  return useContext(AdminAuthContext);
}

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, error, login, authConfig } = useAdminAuth();
  const [apiKey, setApiKey] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);

  const handleLogin = async () => {
    setLoginError(null);
    try {
      await login(apiKey);
    } catch (err) {
      setLoginError(err instanceof Error ? err.message : "Login failed");
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="w-[400px]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="h-5 w-5" />
              Admin Authentication
            </CardTitle>
            <CardDescription>
              Enter your API key to access the admin dashboard
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {(error || loginError) && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error || loginError}</AlertDescription>
              </Alert>
            )}
            <div className="space-y-2">
              <Label htmlFor="apiKey">API Key</Label>
              <Input
                id="apiKey"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your admin API key"
                onKeyDown={(e) => e.key === "Enter" && handleLogin()}
              />
            </div>
            <Button onClick={handleLogin} className="w-full">
              Login
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <AdminAuthContext.Provider value={{ authConfig }}>
      {children}
    </AdminAuthContext.Provider>
  );
}

// =============================================================================
// Feature Flags Tab
// =============================================================================

function FeatureFlagsTab() {
  const { authConfig } = useAdminAuthContext();
  const { techniques, isLoading, error, refresh, toggleTechnique } = useFeatureFlags(authConfig);
  const [newFlagName, setNewFlagName] = useState("");
  const [newFlagEnabled, setNewFlagEnabled] = useState(false);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);

  const handleCreateFlag = async () => {
    if (!newFlagName.trim()) return;
    
    // Note: The current API doesn&apos;t support creating new techniques
    // This would need a backend endpoint to be implemented
    console.warn("Creating new techniques is not yet supported by the API");
    setNewFlagName("");
    setNewFlagEnabled(false);
    setIsCreateDialogOpen(false);
  };

  const handleToggleFlag = async (flagName: string, currentValue: boolean) => {
    try {
      await toggleTechnique(flagName, !currentValue);
    } catch (err) {
      console.error("Failed to toggle technique:", err);
    }
  };

  const handleDeleteFlag = async (flagName: string) => {
    if (!confirm(`Are you sure you want to reset the technique "${flagName}"?`)) return;
    
    // Note: The current API supports reset, not delete
    console.warn("Deleting techniques is not supported - use reset instead");
  };

  if (isLoading) {
    return <div className="flex justify-center p-8"><RefreshCw className="h-6 w-6 animate-spin" /></div>;
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Technique Flags</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Add Flag
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Feature Flag</DialogTitle>
                <DialogDescription>
                  Add a new feature flag to control application behavior.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="flagName">Flag Name</Label>
                  <Input
                    id="flagName"
                    value={newFlagName}
                    onChange={(e) => setNewFlagName(e.target.value)}
                    placeholder="e.g., enable_new_feature"
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="flagEnabled"
                    checked={newFlagEnabled}
                    onCheckedChange={setNewFlagEnabled}
                  />
                  <Label htmlFor="flagEnabled">Enabled by default</Label>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateFlag}>Create</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Technique Name</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {techniques.map((technique) => (
            <TableRow key={technique.name}>
              <TableCell className="font-mono">{technique.name}</TableCell>
              <TableCell>
                <Badge variant={technique.enabled ? "default" : "secondary"}>
                  {technique.enabled ? (
                    <><CheckCircle className="h-3 w-3 mr-1" /> Enabled</>
                  ) : (
                    <><XCircle className="h-3 w-3 mr-1" /> Disabled</>
                  )}
                </Badge>
              </TableCell>
              <TableCell className="text-right">
                <div className="flex justify-end gap-2">
                  <Switch
                    checked={technique.enabled}
                    onCheckedChange={() => handleToggleFlag(technique.name, technique.enabled)}
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteFlag(technique.name)}
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
          {techniques.length === 0 && (
            <TableRow>
              <TableCell colSpan={3} className="text-center text-muted-foreground">
                No techniques configured
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}

// =============================================================================
// Tenants Tab
// =============================================================================

function TenantsTab() {
  const { authConfig } = useAdminAuthContext();
  const { tenants, isLoading, error, refresh, createTenant, deleteTenant } = useTenants(authConfig);
  const [newTenantName, setNewTenantName] = useState("");
  const [newTenantPlan, setNewTenantPlan] = useState("free");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);

  const handleCreateTenant = async () => {
    if (!newTenantName.trim()) return;
    
    try {
      // Generate a tenant_id from the name (lowercase, replace spaces with dashes)
      const tenantId = newTenantName.trim().toLowerCase().replace(/\s+/g, '-');
      await createTenant({
        tenant_id: tenantId,
        name: newTenantName,
        tier: newTenantPlan as "free" | "basic" | "professional" | "enterprise"
      });
      setNewTenantName("");
      setNewTenantPlan("free");
      setIsCreateDialogOpen(false);
    } catch (err) {
      console.error("Failed to create tenant:", err);
    }
  };

  const handleDeleteTenant = async (tenantId: string) => {
    if (!confirm("Are you sure you want to delete this tenant?")) return;
    
    try {
      await deleteTenant(tenantId);
    } catch (err) {
      console.error("Failed to delete tenant:", err);
    }
  };

  if (isLoading) {
    return <div className="flex justify-center p-8"><RefreshCw className="h-6 w-6 animate-spin" /></div>;
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Tenants</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Add Tenant
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Tenant</DialogTitle>
                <DialogDescription>
                  Add a new tenant to the system.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="tenantName">Tenant Name</Label>
                  <Input
                    id="tenantName"
                    value={newTenantName}
                    onChange={(e) => setNewTenantName(e.target.value)}
                    placeholder="e.g., Acme Corp"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="tenantPlan">Plan</Label>
                  <Select value={newTenantPlan} onValueChange={setNewTenantPlan}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a plan" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="free">Free</SelectItem>
                      <SelectItem value="basic">Basic</SelectItem>
                      <SelectItem value="professional">Professional</SelectItem>
                      <SelectItem value="enterprise">Enterprise</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateTenant}>Create</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Plan</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Rate Limit</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {tenants.map((tenant) => (
            <TableRow key={tenant.tenant_id}>
              <TableCell className="font-mono text-xs">{tenant.tenant_id}</TableCell>
              <TableCell>{tenant.name}</TableCell>
              <TableCell>
                <Badge variant="outline">{tenant.tier}</Badge>
              </TableCell>
              <TableCell>
                <Badge variant={tenant.is_active ? "default" : "secondary"}>
                  {tenant.is_active ? "Active" : "Inactive"}
                </Badge>
              </TableCell>
              <TableCell className="text-sm text-muted-foreground">
                {tenant.rate_limit_per_minute}/min
              </TableCell>
              <TableCell className="text-right">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDeleteTenant(tenant.tenant_id)}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </TableCell>
            </TableRow>
          ))}
          {tenants.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} className="text-center text-muted-foreground">
                No tenants found
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}

// =============================================================================
// Analytics Tab
// =============================================================================

function AnalyticsTab() {
  const { authConfig } = useAdminAuthContext();
  const { globalUsage: analytics, isLoading, error, fetchGlobalUsage } = useUsageAnalytics(authConfig);
  const [dateRange, setDateRange] = useState({ start: getDateDaysAgo(7), end: getToday() });

  // Fetch data on mount and when date range changes
  useEffect(() => {
    if (authConfig) {
      fetchGlobalUsage({ start_date: dateRange.start, end_date: dateRange.end });
    }
  }, [authConfig, dateRange, fetchGlobalUsage]);

  const handleDateRangeChange = (start: string, end: string) => {
    setDateRange({ start, end });
  };

  const refresh = () => {
    fetchGlobalUsage({ start_date: dateRange.start, end_date: dateRange.end });
  };

  if (isLoading) {
    return <div className="flex justify-center p-8"><RefreshCw className="h-6 w-6 animate-spin" /></div>;
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Usage Analytics</h3>
        <div className="flex gap-2">
          <Select
            value={`${dateRange.start}_${dateRange.end}`}
            onValueChange={(value) => {
              const [start, end] = value.split("_");
              handleDateRangeChange(start, end);
            }}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select period" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={`${getDateDaysAgo(7)}_${getToday()}`}>Last 7 days</SelectItem>
              <SelectItem value={`${getDateDaysAgo(30)}_${getToday()}`}>Last 30 days</SelectItem>
              <SelectItem value={`${getDateDaysAgo(90)}_${getToday()}`}>Last 90 days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {analytics && (
        <>
          {/* Summary Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{analytics.total_requests.toLocaleString()}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Tokens</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{analytics.total_tokens.toLocaleString()}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Errors</CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{analytics.total_errors.toLocaleString()}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Duration</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{analytics.avg_duration_ms.toFixed(0)}ms</div>
              </CardContent>
            </Card>
          </div>

          {/* Additional Stats */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(analytics.cache_hit_rate * 100).toFixed(1)}%</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Period</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  {analytics.period_start} to {analytics.period_end}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Usage by Endpoint */}
          <Card>
            <CardHeader>
              <CardTitle>Usage by Endpoint</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Endpoint</TableHead>
                    <TableHead className="text-right">Requests</TableHead>
                    <TableHead className="text-right">% of Total</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(analytics.requests_by_endpoint).map(([endpoint, count]) => (
                    <TableRow key={endpoint}>
                      <TableCell className="font-medium font-mono text-sm">{endpoint}</TableCell>
                      <TableCell className="text-right">{count.toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        {((count / analytics.total_requests) * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  ))}
                  {Object.keys(analytics.requests_by_endpoint).length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} className="text-center text-muted-foreground">
                        No endpoint data available
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Usage by Technique */}
          <Card>
            <CardHeader>
              <CardTitle>Usage by Technique</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Technique</TableHead>
                    <TableHead className="text-right">Requests</TableHead>
                    <TableHead className="text-right">% of Total</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(analytics.requests_by_technique).map(([technique, count]) => (
                    <TableRow key={technique}>
                      <TableCell className="font-medium">{technique}</TableCell>
                      <TableCell className="text-right">{count.toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        {((count / analytics.total_requests) * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  ))}
                  {Object.keys(analytics.requests_by_technique).length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} className="text-center text-muted-foreground">
                        No technique data available
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

// Helper functions for date handling
function getToday(): string {
  return new Date().toISOString().split("T")[0];
}

function getDateDaysAgo(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().split("T")[0];
}

// =============================================================================
// Main Admin Page
// =============================================================================

export default function AdminPage() {
  return (
    <AuthGuard>
      <div className="container mx-auto py-8 px-4">
        <div className="mb-8">
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="h-8 w-8" />
            Admin Dashboard
          </h1>
          <p className="text-muted-foreground mt-2">
            Manage system settings, feature flags, tenants, and view analytics.
          </p>
        </div>

        <Tabs defaultValue="flags" className="space-y-4">
          <TabsList>
            <TabsTrigger value="flags" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Feature Flags
            </TabsTrigger>
            <TabsTrigger value="tenants" className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              Tenants
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Analytics
            </TabsTrigger>
          </TabsList>

          <TabsContent value="flags">
            <Card>
              <CardContent className="pt-6">
                <FeatureFlagsTab />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="tenants">
            <Card>
              <CardContent className="pt-6">
                <TenantsTab />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics">
            <AnalyticsTab />
          </TabsContent>
        </Tabs>
      </div>
    </AuthGuard>
  );
}