"use client";

import { usePathname } from "next/navigation";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { ConnectionStatus } from "@/components/connection-status";
import { ModelDropdown } from "@/components/model-selector";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Menu, ShieldAlert } from "lucide-react";
import { Sidebar } from "./sidebar";
import { UserMenu } from "./UserMenu";
import { useState } from "react";

export function Header() {
  const pathname = usePathname();
  const segments = pathname.split("/").filter((item) => item !== "");
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-4 border-b bg-background px-4 md:px-6">
      <div className="flex items-center gap-2 md:hidden">
        <Sheet open={isOpen} onOpenChange={setIsOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden">
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle Menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-72">
            <SheetHeader className="p-4 border-b">
              <SheetTitle className="flex items-center gap-2">
                <ShieldAlert className="h-6 w-6 text-primary" />
                <span className="gradient-text font-bold">Chimera</span>
              </SheetTitle>
            </SheetHeader>
            <Sidebar showLogo={false} onNavigate={() => setIsOpen(false)} className="border-r-0" />
          </SheetContent>
        </Sheet>
      </div>

      <div className="flex-1 overflow-hidden">
        <Breadcrumb className="hidden sm:block">
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href="/">Home</BreadcrumbLink>
            </BreadcrumbItem>
            {segments.map((segment, index) => {
              const href = `/${segments.slice(0, index + 1).join("/")}`;
              const isLast = index === segments.length - 1;

              return (
                <div key={href} className="flex items-center">
                  <BreadcrumbSeparator />
                  <BreadcrumbItem className="ml-2 capitalize">
                    {isLast ? (
                      <BreadcrumbPage className="truncate max-w-[150px]">{segment}</BreadcrumbPage>
                    ) : (
                      <BreadcrumbLink href={href} className="truncate max-w-[100px]">{segment}</BreadcrumbLink>
                    )}
                  </BreadcrumbItem>
                </div>
              );
            })}
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="flex items-center gap-2 md:gap-4">
        <div className="hidden xs:block">
          <ModelDropdown />
        </div>
        <ConnectionStatus />
        <UserMenu />
      </div>
    </header>
  );
}
