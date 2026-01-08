"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-enhanced";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Shield,
  FileText,
  AlertTriangle,
  CheckCircle,
  Clock,
  Search,
  Download,
  Activity,
  User,
  Fingerprint
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

// Types for Audit Log
interface AuditEntry {
  timestamp: string;
  action: string;
  severity: "info" | "warning" | "error" | "critical" | "debug";
  user_id: string;
  resource: string;
  ip_address?: string;
  hash: string;
  details: Record<string, any>;
}

interface AuditStats {
  total_events: number;
  events_by_severity: Record<string, number>;
  events_by_action: Record<string, number>;
  last_verification: string;
}

export function ComplianceDashboard() {
  const [filterAction, setFilterAction] = useState<string>("all");
  const [filterSeverity, setFilterSeverity] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [verifying, setVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<any>(null);

  // Fetch Audit Logs
  const { data: logsData, isLoading: logsLoading, refetch: refetchLogs } = useQuery({
    queryKey: ["audit-logs", filterAction, filterSeverity],
    queryFn: async () => {
      const params: Record<string, any> = { limit: 100 };
      if (filterAction !== "all") params.action = filterAction;
      if (filterSeverity !== "all") params.severity = filterSeverity;
      
      const res = await apiClient.get("/audit/logs", { params });
      return res.data;
    },
    refetchInterval: 30000,
  });

  // Fetch Stats
  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ["audit-stats"],
    queryFn: async () => {
      const res = await apiClient.get("/audit/stats");
      return res.data as AuditStats;
    },
    refetchInterval: 60000,
  });

  const handleVerifyChain = async () => {
    setVerifying(true);
    try {
      const res = await apiClient.post("/audit/verify");
      setVerificationResult(res.data);
      setTimeout(() => setVerificationResult(null), 5000);
    } catch (error) {
      setVerificationResult({ status: "error", message: "Verification Failed" });
    } finally {
      setVerifying(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical": return "destructive";
      case "error": return "destructive";
      case "warning": return "warning"; // Assuming warning variant exists, else "secondary"
      case "info": return "default"; // or "secondary"
      case "debug": return "outline";
      default: return "secondary";
    }
  };

  const filteredLogs = logsData?.logs?.filter((log: AuditEntry) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      log.action.toLowerCase().includes(query) ||
      log.user_id?.toLowerCase().includes(query) ||
      log.resource.toLowerCase().includes(query)
    );
  }) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Compliance & Audit</h2>
          <p className="text-muted-foreground">
            System-wide audit trails and integrity verification.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => handleVerifyChain()} disabled={verifying}>
            {verifying ? <Clock className="mr-2 h-4 w-4 animate-spin" /> : <Shield className="mr-2 h-4 w-4" />}
            {verifying ? "Verifying Chain..." : "Verify Integrity"}
          </Button>
          <Button variant="default">
            <Download className="mr-2 h-4 w-4" /> Export Report
          </Button>
        </div>
      </div>

      {/* Verification Result Toast/Banner */}
      {verificationResult && (
        <div className={cn(
          "p-4 rounded-lg border flex items-center gap-3",
          verificationResult.status === "success" 
            ? "bg-green-500/10 border-green-500/20 text-green-700 dark:text-green-400"
            : "bg-red-500/10 border-red-500/20 text-red-700 dark:text-red-400"
        )}>
          {verificationResult.status === "success" ? (
            <CheckCircle className="h-5 w-5" />
          ) : (
            <AlertTriangle className="h-5 w-5" />
          )}
          <div>
            <p className="font-semibold">{verificationResult.message}</p>
            {verificationResult.total_verified && (
              <p className="text-xs opacity-90">Verified {verificationResult.total_verified} chained entries.</p>
            )}
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Events</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statsData?.total_events || 0}</div>
            <p className="text-xs text-muted-foreground mt-1">Logged system actions</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Security Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {(statsData?.events_by_severity?.critical || 0) + (statsData?.events_by_severity?.error || 0)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Critical/Error events</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Verified</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg font-medium">
              {statsData?.last_verification 
                ? format(new Date(statsData.last_verification), "MMM d, HH:mm") 
                : "Never"}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Chain integrity check</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Users</CardTitle>
            <User className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">--</div>
            <p className="text-xs text-muted-foreground mt-1">Unique actors today</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Audit Log Search</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search actions, resources, or users..."
                className="pl-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Select value={filterSeverity} onValueChange={setFilterSeverity}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="error">Error</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterAction} onValueChange={setFilterAction}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Action Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Actions</SelectItem>
                <SelectItem value="auth.login">Login</SelectItem>
                <SelectItem value="prompt.jailbreak">Jailbreak</SelectItem>
                <SelectItem value="config.change">Config Change</SelectItem>
                <SelectItem value="security.blocked_request">Blocked Request</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Logs Table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[180px]">Timestamp</TableHead>
                <TableHead className="w-[100px]">Severity</TableHead>
                <TableHead className="w-[180px]">Action</TableHead>
                <TableHead className="w-[150px]">User / IP</TableHead>
                <TableHead>Resource & Details</TableHead>
                <TableHead className="w-[80px] text-right">Hash</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {logsLoading ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    Loading audit logs...
                  </TableCell>
                </TableRow>
              ) : filteredLogs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    No audit records found matching your filters.
                  </TableCell>
                </TableRow>
              ) : (
                filteredLogs.map((log: AuditEntry) => (
                  <TableRow key={log.hash}>
                    <TableCell className="font-mono text-xs">
                      {format(new Date(log.timestamp), "yyyy-MM-dd HH:mm:ss")}
                    </TableCell>
                    <TableCell>
                      <Badge variant={getSeverityColor(log.severity) as any} className="capitalize">
                        {log.severity}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-medium">{log.action}</TableCell>
                    <TableCell>
                      <div className="flex flex-col text-xs">
                        <span className="font-medium">{log.user_id || "System"}</span>
                        <span className="text-muted-foreground">{log.ip_address || "-"}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1">
                        <span className="font-medium text-sm">{log.resource}</span>
                        <span className="text-xs text-muted-foreground truncate max-w-[400px]">
                          {JSON.stringify(log.details)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end" title={log.hash}>
                        <Fingerprint className="h-4 w-4 text-muted-foreground hover:text-primary cursor-help" />
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}