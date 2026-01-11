"use client";

/**
 * User Profile Page
 *
 * Comprehensive profile management page with tabs for:
 * - Profile information (view and edit)
 * - Password change
 * - API key management
 * - Activity log preview
 *
 * @module app/dashboard/profile/page
 */

import React from "react";
import Link from "next/link";
import {
  User,
  Lock,
  Key,
  Activity,
  ArrowLeft,
  Shield,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  TooltipProvider,
} from "@/components/ui/tooltip";

import { ProfileForm } from "@/components/profile/ProfileForm";
import { ChangePasswordForm } from "@/components/profile/ChangePasswordForm";
import { APIKeyManager } from "@/components/profile/APIKeyManager";

// =============================================================================
// Activity Log Preview Component
// =============================================================================

function ActivityLogPreview() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h3 className="text-lg font-semibold">Recent Activity</h3>
          <p className="text-sm text-muted-foreground">
            Your recent account activity and security events
          </p>
        </div>
        <Link href="/dashboard/activity">
          <Button variant="outline" className="gap-2">
            <Activity className="h-4 w-4" />
            View All Activity
          </Button>
        </Link>
      </div>

      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <Activity className="h-12 w-12 text-muted-foreground/50 mb-4" />
          <h3 className="text-lg font-medium mb-2">Activity Log</h3>
          <p className="text-sm text-muted-foreground mb-4">
            View your complete activity history including logins,
            <br />
            password changes, and API key operations.
          </p>
          <Link href="/dashboard/activity">
            <Button variant="secondary" className="gap-2">
              <Activity className="h-4 w-4" />
              Go to Activity Log
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// Security Tips Component
// =============================================================================

function SecurityTips() {
  const tips = [
    {
      icon: Lock,
      title: "Use a Strong Password",
      description:
        "At least 12 characters with a mix of letters, numbers, and symbols.",
    },
    {
      icon: Key,
      title: "Rotate API Keys",
      description:
        "Regularly rotate your API keys and revoke any that are unused.",
    },
    {
      icon: Shield,
      title: "Monitor Activity",
      description:
        "Regularly check your activity log for any suspicious access.",
    },
  ];

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Shield className="h-4 w-4 text-primary" />
          Security Tips
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {tips.map((tip, index) => {
          const Icon = tip.icon;
          return (
            <div key={index} className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <Icon className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">{tip.title}</p>
                <p className="text-xs text-muted-foreground">
                  {tip.description}
                </p>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Profile Page Component
// =============================================================================

export default function ProfilePage() {
  return (
    <TooltipProvider>
      <div className="flex flex-col gap-6 p-6">
        {/* Page Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Link href="/dashboard">
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              </Link>
              <User className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-bold tracking-tight">Profile</h1>
            </div>
            <p className="text-muted-foreground ml-10">
              Manage your account settings and security preferences
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Profile Content - Takes 2 columns on large screens */}
          <div className="lg:col-span-2">
            <Tabs defaultValue="profile" className="w-full">
              <TabsList className="grid w-full grid-cols-4 max-w-lg">
                <TabsTrigger value="profile" className="gap-2">
                  <User className="h-4 w-4 hidden sm:inline-block" />
                  Profile
                </TabsTrigger>
                <TabsTrigger value="password" className="gap-2">
                  <Lock className="h-4 w-4 hidden sm:inline-block" />
                  Password
                </TabsTrigger>
                <TabsTrigger value="api-keys" className="gap-2">
                  <Key className="h-4 w-4 hidden sm:inline-block" />
                  API Keys
                </TabsTrigger>
                <TabsTrigger value="activity" className="gap-2">
                  <Activity className="h-4 w-4 hidden sm:inline-block" />
                  Activity
                </TabsTrigger>
              </TabsList>

              {/* Profile Tab */}
              <TabsContent value="profile" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Profile Information</CardTitle>
                    <CardDescription>
                      View and update your account details
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ProfileForm />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Password Tab */}
              <TabsContent value="password" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Change Password</CardTitle>
                    <CardDescription>
                      Update your password to keep your account secure
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ChangePasswordForm />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* API Keys Tab */}
              <TabsContent value="api-keys" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>API Key Management</CardTitle>
                    <CardDescription>
                      Create and manage API keys for programmatic access
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <APIKeyManager />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Activity Tab */}
              <TabsContent value="activity" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Activity Log</CardTitle>
                    <CardDescription>
                      Monitor your recent account activity
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ActivityLogPreview />
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Security Tips */}
            <SecurityTips />

            {/* Quick Actions */}
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-sm font-medium">
                  Quick Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Link href="/dashboard/settings" className="block">
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    size="sm"
                  >
                    Platform Settings
                  </Button>
                </Link>
                <Link href="/dashboard/activity" className="block">
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    size="sm"
                  >
                    View Full Activity Log
                  </Button>
                </Link>
              </CardContent>
            </Card>

            {/* Support */}
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-sm font-medium">
                  Need Help?
                </CardTitle>
                <CardDescription>
                  Contact support for account-related issues
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  For email changes, account recovery, or other issues, please
                  contact your system administrator.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
