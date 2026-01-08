"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Home, ArrowLeft, Search, BookOpen } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-background to-muted/20 p-4">
      <Card className="w-full max-w-lg text-center">
        <CardHeader className="pb-4">
          <div className="mx-auto mb-4 text-muted-foreground">
            <Search className="h-16 w-16" />
          </div>
          <CardTitle className="text-3xl font-bold">404 - Page Not Found</CardTitle>
          <CardDescription className="text-lg">
            The page you&apos;re looking for doesn&apos;t exist or has been moved.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="text-sm text-muted-foreground">
            <p>You might want to try one of these:</p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link href="/dashboard">
              <Button className="w-full sm:w-auto">
                <Home className="mr-2 h-4 w-4" />
                Go to Dashboard
              </Button>
            </Link>
            
            <Button 
              variant="outline" 
              onClick={() => window.history.back()}
              className="w-full sm:w-auto"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Go Back
            </Button>
          </div>
          
          <div className="pt-4 border-t">
            <p className="text-sm text-muted-foreground mb-3">Available Routes:</p>
            <div className="flex flex-wrap gap-2 justify-center">
              <Link href="/dashboard">
                <Button variant="ghost" size="sm">/dashboard</Button>
              </Link>
              <Link href="/dashboard/jailbreak">
                <Button variant="ghost" size="sm">/dashboard/jailbreak</Button>
              </Link>
              <Link href="/dashboard/gptfuzz">
                <Button variant="ghost" size="sm">/dashboard/gptfuzz</Button>
              </Link>
            </div>
          </div>
          
          <div className="pt-4 text-xs text-muted-foreground">
            <p>If you believe this is an error, check the</p>
            <Link 
              href="https://github.com/your-repo/chimera" 
              className="text-primary hover:underline inline-flex items-center gap-1"
            >
              <BookOpen className="h-3 w-3" />
              documentation
            </Link>
            <span> or report an issue.</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
