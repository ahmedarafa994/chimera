"use client";

/**
 * UserMenu Component
 *
 * Displays user avatar, name, and role badge in the header with a dropdown menu
 * for profile, settings, and logout actions.
 *
 * @module components/layout/UserMenu
 */

import { useRouter } from "next/navigation";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuGroup,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import {
  User,
  Settings,
  LogOut,
  Shield,
  FlaskConical,
  Eye,
  ChevronDown,
  Activity,
  Key,
} from "lucide-react";
import { useAuth, type UserRole } from "@/hooks/useAuth";

// =============================================================================
// Role Configuration
// =============================================================================

interface RoleConfig {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badgeClassName: string;
}

const ROLE_CONFIG: Record<UserRole, RoleConfig> = {
  admin: {
    label: "Admin",
    icon: Shield,
    badgeClassName:
      "bg-gradient-to-r from-red-500/20 to-orange-500/20 text-orange-400 border-orange-500/30",
  },
  researcher: {
    label: "Researcher",
    icon: FlaskConical,
    badgeClassName:
      "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-cyan-400 border-cyan-500/30",
  },
  viewer: {
    label: "Viewer",
    icon: Eye,
    badgeClassName:
      "bg-gradient-to-r from-gray-500/20 to-slate-500/20 text-slate-400 border-slate-500/30",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate avatar initials from username or email
 */
function getInitials(username?: string, email?: string): string {
  if (username) {
    // Take first two letters of username
    return username.slice(0, 2).toUpperCase();
  }
  if (email) {
    // Take first letter before @ and first letter after
    const parts = email.split("@");
    if (parts.length === 2 && parts[0] && parts[1]) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return email.slice(0, 2).toUpperCase();
  }
  return "??";
}

/**
 * Generate a consistent color based on username/email for avatar background
 */
function getAvatarColor(username?: string): string {
  if (!username) return "bg-primary";

  // Simple hash function to generate consistent color
  let hash = 0;
  for (let i = 0; i < username.length; i++) {
    hash = username.charCodeAt(i) + ((hash << 5) - hash);
  }

  const colors = [
    "bg-gradient-to-br from-violet-500 to-purple-600",
    "bg-gradient-to-br from-blue-500 to-cyan-600",
    "bg-gradient-to-br from-emerald-500 to-teal-600",
    "bg-gradient-to-br from-orange-500 to-amber-600",
    "bg-gradient-to-br from-pink-500 to-rose-600",
    "bg-gradient-to-br from-indigo-500 to-blue-600",
  ];

  return colors[Math.abs(hash) % colors.length] || "bg-primary";
}

// =============================================================================
// Loading State Component
// =============================================================================

function UserMenuSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <Skeleton className="h-8 w-8 rounded-full" />
      <div className="hidden lg:flex flex-col gap-1">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-14" />
      </div>
    </div>
  );
}

// =============================================================================
// Role Badge Component
// =============================================================================

interface RoleBadgeProps {
  role: UserRole;
  className?: string;
}

function RoleBadge({ role, className = "" }: RoleBadgeProps) {
  const config = ROLE_CONFIG[role];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={`${config.badgeClassName} ${className} text-[10px] px-1.5 py-0 h-4`}
    >
      <Icon className="h-2.5 w-2.5 mr-0.5" />
      {config.label}
    </Badge>
  );
}

// =============================================================================
// UserMenu Component
// =============================================================================

export interface UserMenuProps {
  /** Optional className for the container */
  className?: string;
  /** Whether to show the expanded version with username */
  expanded?: boolean;
}

export function UserMenu({ className = "", expanded = true }: UserMenuProps) {
  const router = useRouter();
  const { user, isLoading, isAuthenticated, logout } = useAuth();

  // Show loading skeleton
  if (isLoading) {
    return <UserMenuSkeleton />;
  }

  // If not authenticated, don't render anything
  // (ProtectedRoute should handle redirect)
  if (!isAuthenticated || !user) {
    return null;
  }

  const initials = getInitials(user.username, user.email);
  const avatarColor = getAvatarColor(user.username);

  const handleLogout = async () => {
    try {
      await logout();
      router.push("/login");
    } catch {
      // Error handled by auth context
      router.push("/login");
    }
  };

  const handleNavigate = (path: string) => {
    router.push(path);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={`flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-accent focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background ${className}`}
          aria-label="User menu"
        >
          <Avatar className="h-8 w-8 ring-2 ring-background shadow-md">
            <AvatarFallback className={`${avatarColor} text-white text-xs font-semibold`}>
              {initials}
            </AvatarFallback>
          </Avatar>

          {expanded && (
            <>
              <div className="hidden lg:flex flex-col items-start">
                <span className="text-sm font-medium text-foreground leading-tight">
                  {user.username}
                </span>
                <RoleBadge role={user.role} />
              </div>
              <ChevronDown className="hidden lg:block h-4 w-4 text-muted-foreground" />
            </>
          )}
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-64">
        {/* User Info Header */}
        <DropdownMenuLabel className="p-4">
          <div className="flex items-center gap-3">
            <Avatar className="h-12 w-12 ring-2 ring-primary/20 shadow-lg">
              <AvatarFallback className={`${avatarColor} text-white text-sm font-semibold`}>
                {initials}
              </AvatarFallback>
            </Avatar>
            <div className="flex flex-col">
              <span className="font-semibold text-foreground">{user.username}</span>
              <span className="text-xs text-muted-foreground truncate max-w-[140px]">
                {user.email}
              </span>
              <RoleBadge role={user.role} className="mt-1" />
            </div>
          </div>
        </DropdownMenuLabel>

        <DropdownMenuSeparator />

        {/* Navigation Items */}
        <DropdownMenuGroup>
          <DropdownMenuItem
            onClick={() => handleNavigate("/dashboard/profile")}
            className="cursor-pointer"
          >
            <User className="mr-2 h-4 w-4" />
            <span>Profile</span>
          </DropdownMenuItem>

          <DropdownMenuItem
            onClick={() => handleNavigate("/dashboard/profile")}
            className="cursor-pointer"
          >
            <Key className="mr-2 h-4 w-4" />
            <span>API Keys</span>
          </DropdownMenuItem>

          <DropdownMenuItem
            onClick={() => handleNavigate("/dashboard/activity")}
            className="cursor-pointer"
          >
            <Activity className="mr-2 h-4 w-4" />
            <span>Activity Log</span>
          </DropdownMenuItem>

          <DropdownMenuItem
            onClick={() => handleNavigate("/dashboard/settings")}
            className="cursor-pointer"
          >
            <Settings className="mr-2 h-4 w-4" />
            <span>Settings</span>
          </DropdownMenuItem>
        </DropdownMenuGroup>

        <DropdownMenuSeparator />

        {/* Logout */}
        <DropdownMenuItem
          onClick={handleLogout}
          className="cursor-pointer text-destructive focus:text-destructive focus:bg-destructive/10"
        >
          <LogOut className="mr-2 h-4 w-4" />
          <span>Log out</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default UserMenu;
