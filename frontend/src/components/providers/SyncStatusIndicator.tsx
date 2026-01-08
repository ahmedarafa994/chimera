'use client';

/**
 * Sync Status Indicator Component
 * 
 * Displays the current synchronization status with visual indicators
 * and provides controls for manual sync operations.
 */

import React from 'react';
import { RefreshCw, Wifi, WifiOff, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { SyncStatus } from '@/types/provider-sync';

interface SyncStatusIndicatorProps {
  status: SyncStatus;
  isConnected: boolean;
  isLoading?: boolean;
  lastSyncTime?: Date;
  version?: number;
  error?: string;
  onSync?: () => void;
  className?: string;
  showDetails?: boolean;
  compact?: boolean;
}

const statusConfig: Record<SyncStatus, {
  icon: React.ElementType;
  label: string;
  color: string;
  bgColor: string;
}> = {
  [SyncStatus.SYNCED]: {
    icon: CheckCircle,
    label: 'Synced',
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-500/10',
  },
  [SyncStatus.SYNCING]: {
    icon: RefreshCw,
    label: 'Syncing',
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
  },
  [SyncStatus.STALE]: {
    icon: Clock,
    label: 'Stale',
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/10',
  },
  [SyncStatus.ERROR]: {
    icon: AlertCircle,
    label: 'Error',
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
  },
  [SyncStatus.DISCONNECTED]: {
    icon: WifiOff,
    label: 'Disconnected',
    color: 'text-zinc-500',
    bgColor: 'bg-zinc-500/10',
  },
};

export function SyncStatusIndicator({
  status,
  isConnected,
  isLoading = false,
  lastSyncTime,
  version,
  error,
  onSync,
  className,
  showDetails = false,
  compact = false,
}: SyncStatusIndicatorProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  const formatLastSync = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  };

  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                'flex items-center gap-1.5 px-2 py-1 rounded-full',
                config.bgColor,
                className
              )}
            >
              <Icon
                className={cn(
                  'h-3.5 w-3.5',
                  config.color,
                  status === SyncStatus.SYNCING && 'animate-spin'
                )}
              />
              {isConnected ? (
                <Wifi className="h-3 w-3 text-emerald-500" />
              ) : (
                <WifiOff className="h-3 w-3 text-zinc-500" />
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-xs">
            <div className="space-y-1 text-xs">
              <p className="font-medium">{config.label}</p>
              {lastSyncTime && (
                <p className="text-muted-foreground">
                  Last sync: {formatLastSync(lastSyncTime)}
                </p>
              )}
              {version !== undefined && (
                <p className="text-muted-foreground">Version: {version}</p>
              )}
              {error && <p className="text-red-400">{error}</p>}
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <div
      className={cn(
        'flex items-center gap-3 p-3 rounded-lg border',
        config.bgColor,
        'border-border/50',
        className
      )}
    >
      <div className="flex items-center gap-2">
        <Icon
          className={cn(
            'h-4 w-4',
            config.color,
            status === SyncStatus.SYNCING && 'animate-spin'
          )}
        />
        <span className={cn('text-sm font-medium', config.color)}>
          {config.label}
        </span>
      </div>

      <div className="flex items-center gap-2 ml-auto">
        {/* Connection indicator */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1">
                {isConnected ? (
                  <Wifi className="h-4 w-4 text-emerald-500" />
                ) : (
                  <WifiOff className="h-4 w-4 text-zinc-500" />
                )}
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {isConnected ? 'Real-time updates active' : 'Disconnected - using polling'}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* Version badge */}
        {version !== undefined && showDetails && (
          <Badge variant="outline" className="text-xs">
            v{version}
          </Badge>
        )}

        {/* Last sync time */}
        {lastSyncTime && showDetails && (
          <span className="text-xs text-muted-foreground">
            {formatLastSync(lastSyncTime)}
          </span>
        )}

        {/* Sync button */}
        {onSync && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onSync}
            disabled={isLoading || status === SyncStatus.SYNCING}
            className="h-7 px-2"
          >
            <RefreshCw
              className={cn(
                'h-3.5 w-3.5',
                (isLoading || status === SyncStatus.SYNCING) && 'animate-spin'
              )}
            />
          </Button>
        )}
      </div>

      {/* Error message */}
      {error && showDetails && (
        <div className="w-full mt-2 text-xs text-red-400">{error}</div>
      )}
    </div>
  );
}

// =============================================================================
// Model Deprecation Warning
// =============================================================================

interface ModelDeprecationWarningProps {
  modelId: string;
  modelName: string;
  deprecationDate?: string;
  sunsetDate?: string;
  replacementModelId?: string;
  replacementModelName?: string;
  onSelectReplacement?: (modelId: string) => void;
  className?: string;
}

export function ModelDeprecationWarning({
  modelId,
  modelName,
  deprecationDate,
  sunsetDate,
  replacementModelId,
  replacementModelName,
  onSelectReplacement,
  className,
}: ModelDeprecationWarningProps) {
  const formatRelativeTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays < 0) return 'past due';
    if (diffDays === 0) return 'today';
    if (diffDays === 1) return 'tomorrow';
    if (diffDays < 7) return `in ${diffDays} days`;
    if (diffDays < 30) return `in ${diffDays} days`;

    return new Intl.DateTimeFormat('en-US', {
      dateStyle: 'medium',
      timeStyle: 'short',
      day: 'numeric',
    }).format(date);
  };

  const formatDate = (dateStr: string) => {
    return new Intl.DateTimeFormat('en-US', {
      dateStyle: 'medium',
      timeStyle: 'short',
      day: 'numeric',
    }).format(new Date(dateStr));
  };

  const isUrgent = useMemo(() => {
    return sunsetDate && new Date(sunsetDate).getTime() - Date.now() < 30 * 24 * 60 * 60 * 1000;
  }, [sunsetDate]);

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-3 rounded-lg border',
        isUrgent
          ? 'bg-red-500/10 border-red-500/30'
          : 'bg-amber-500/10 border-amber-500/30',
        className
      )}
    >
      <AlertCircle
        className={cn(
          'h-5 w-5 mt-0.5 flex-shrink-0',
          isUrgent ? 'text-red-500' : 'text-amber-500'
        )}
      />
      
      <div className="flex-1 space-y-1">
        <p className={cn('text-sm font-medium', isUrgent ? 'text-red-400' : 'text-amber-400')}>
          {modelName} is deprecated
        </p>
        
        <div className="text-xs text-muted-foreground space-y-0.5">
          {deprecationDate && (
            <p>Deprecated on: {formatDate(deprecationDate)}</p>
          )}
          {sunsetDate && (
            <p className={isUrgent ? 'text-red-400 font-medium' : ''}>
              Will be removed: {formatDate(sunsetDate)}
            </p>
          )}
        </div>

        {replacementModelId && (
          <div className="mt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onSelectReplacement?.(replacementModelId)}
              className="h-7 text-xs"
            >
              Switch to {replacementModelName || replacementModelId}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Provider Unavailable Warning
// =============================================================================

interface ProviderUnavailableWarningProps {
  providerId: string;
  providerName: string;
  status: string;
  errorMessage?: string;
  fallbackProviderId?: string;
  fallbackProviderName?: string;
  estimatedRecoveryTime?: string;
  onSelectFallback?: (providerId: string) => void;
  className?: string;
}

export function ProviderUnavailableWarning({
  providerId,
  providerName,
  status,
  errorMessage,
  fallbackProviderId,
  fallbackProviderName,
  estimatedRecoveryTime,
  onSelectFallback,
  className,
}: ProviderUnavailableWarningProps) {
  return (
    <div
      className={cn(
        'flex items-start gap-3 p-3 rounded-lg border',
        'bg-red-500/10 border-red-500/30',
        className
      )}
    >
      <WifiOff className="h-5 w-5 mt-0.5 flex-shrink-0 text-red-500" />
      
      <div className="flex-1 space-y-1">
        <p className="text-sm font-medium text-red-400">
          {providerName} is unavailable
        </p>
        
        <div className="text-xs text-muted-foreground space-y-0.5">
          <p>Status: {status}</p>
          {errorMessage && <p>{errorMessage}</p>}
          {estimatedRecoveryTime && (
            <p>Estimated recovery: {estimatedRecoveryTime}</p>
          )}
        </div>

        {fallbackProviderId && (
          <div className="mt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onSelectFallback?.(fallbackProviderId)}
              className="h-7 text-xs"
            >
              Use {fallbackProviderName || fallbackProviderId} instead
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}