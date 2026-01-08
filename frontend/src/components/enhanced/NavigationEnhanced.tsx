'use client';

/**
 * Enhanced Navigation Components
 *
 * Provides navigation UI elements:
 * - Breadcrumbs with animations
 * - Tab navigation with indicators
 * - Pagination with smart display
 * - Step indicators
 * - Command palette
 *
 * Built on Shadcn/Radix primitives
 */

import * as React from 'react';
import { forwardRef, useState, useCallback, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import {
  ChevronLeft,
  ChevronRight,
  Home,
  Search,
  Command,
  ArrowRight,
  Check,
  Circle,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ReactNode;
}

export interface TabItem {
  value: string;
  label: string;
  icon?: React.ReactNode;
  badge?: string | number;
  disabled?: boolean;
}

export interface StepItem {
  label: string;
  description?: string;
  icon?: React.ReactNode;
}

export interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  shortcut?: string;
  action: () => void;
  group?: string;
}

// ============================================================================
// Enhanced Breadcrumbs
// ============================================================================

export interface EnhancedBreadcrumbsProps extends React.HTMLAttributes<HTMLElement> {
  /** Breadcrumb items */
  items: BreadcrumbItem[];
  /** Show home icon */
  showHome?: boolean;
  /** Home href */
  homeHref?: string;
  /** Max items to show (rest collapsed) */
  maxItems?: number;
  /** Separator */
  separator?: React.ReactNode;
}

export const EnhancedBreadcrumbs = forwardRef<HTMLElement, EnhancedBreadcrumbsProps>(
  function EnhancedBreadcrumbs(
    {
      className,
      items,
      showHome = true,
      homeHref = '/',
      maxItems = 4,
      separator,
      ...props
    },
    ref
  ) {
    const shouldCollapse = items.length > maxItems;
    const visibleItems = shouldCollapse
      ? [items[0], ...items.slice(-2)]
      : items;

    return (
      <Breadcrumb ref={ref} className={className} {...props}>
        <BreadcrumbList>
          {showHome && (
            <>
              <BreadcrumbItem>
                <BreadcrumbLink asChild>
                  <Link
                    href={homeHref as unknown as "/"}
                    className="flex items-center gap-1 hover:text-foreground transition-colors"
                  >
                    <Home className="h-4 w-4" />
                    <span className="sr-only">Home</span>
                  </Link>
                </BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator>{separator}</BreadcrumbSeparator>
            </>
          )}

          {visibleItems.map((item, index) => {
            const isLast = index === visibleItems.length - 1;
            const showEllipsis = shouldCollapse && index === 0;

            return (
              <React.Fragment key={item.label}>
                {showEllipsis && index > 0 && (
                  <>
                    <BreadcrumbItem>
                      <span className="text-muted-foreground">...</span>
                    </BreadcrumbItem>
                    <BreadcrumbSeparator>{separator}</BreadcrumbSeparator>
                  </>
                )}

                <BreadcrumbItem
                  className={cn(
                    'animate-in fade-in slide-in-from-left-2 duration-200',
                    { 'animation-delay-100': index === 1 },
                    { 'animation-delay-200': index === 2 }
                  )}
                >
                  {isLast ? (
                    <BreadcrumbPage className="flex items-center gap-1.5 font-medium">
                      {item.icon}
                      {item.label}
                    </BreadcrumbPage>
                  ) : (
                    <BreadcrumbLink asChild>
                      <Link
                        href={(item.href || '#') as unknown as "/"}
                        className="flex items-center gap-1.5 hover:text-foreground transition-colors"
                      >
                        {item.icon}
                        {item.label}
                      </Link>
                    </BreadcrumbLink>
                  )}
                </BreadcrumbItem>

                {!isLast && <BreadcrumbSeparator>{separator}</BreadcrumbSeparator>}
              </React.Fragment>
            );
          })}
        </BreadcrumbList>
      </Breadcrumb>
    );
  }
);

// ============================================================================
// Enhanced Tabs
// ============================================================================

export interface EnhancedTabsProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Tab items */
  tabs: TabItem[];
  /** Default value */
  defaultValue?: string;
  /** Controlled value */
  value?: string;
  /** On value change */
  onValueChange?: (value: string) => void;
  /** Variant */
  variant?: 'default' | 'pills' | 'underline';
  /** Size */
  size?: 'sm' | 'md' | 'lg';
  /** Full width */
  fullWidth?: boolean;
  /** Children (tab content) */
  children?: React.ReactNode;
}

const tabVariants = {
  default: '',
  pills: 'bg-muted p-1 rounded-lg',
  underline: 'border-b border-border',
};

const tabTriggerVariants = {
  default: 'data-[state=active]:bg-background data-[state=active]:shadow-sm',
  pills: 'rounded-md data-[state=active]:bg-background data-[state=active]:shadow-sm',
  underline: 'rounded-none border-b-2 border-transparent data-[state=active]:border-primary',
};

const tabSizes = {
  sm: 'h-8 text-xs px-3',
  md: 'h-9 text-sm px-4',
  lg: 'h-10 text-base px-5',
};

export const EnhancedTabs = forwardRef<HTMLDivElement, EnhancedTabsProps>(
  function EnhancedTabs(
    {
      className,
      tabs,
      defaultValue,
      value,
      onValueChange,
      variant = 'default',
      size = 'md',
      fullWidth = false,
      children,
      ...props
    },
    ref
  ) {
    const [activeTab, setActiveTab] = useState(value || defaultValue || tabs[0]?.value);
    const [indicatorStyle, setIndicatorStyle] = useState<React.CSSProperties>({});
    const tabsRef = useRef<HTMLDivElement>(null);

    // Update indicator position
    useEffect(() => {
      if (variant !== 'underline' || !tabsRef.current) return;

      const activeElement = tabsRef.current.querySelector(
        `[data-state="active"]`
      ) as HTMLElement;

      if (activeElement) {
        setIndicatorStyle({
          left: activeElement.offsetLeft,
          width: activeElement.offsetWidth,
        });
      }
    }, [activeTab, variant]);

    const handleValueChange = (newValue: string) => {
      setActiveTab(newValue);
      onValueChange?.(newValue);
    };

    // Extract dir from props to cast it properly for Radix Tabs
    const { dir, ...restProps } = props;

    return (
      <Tabs
        ref={ref}
        value={value || activeTab}
        onValueChange={handleValueChange}
        className={className}
        dir={dir as 'ltr' | 'rtl' | undefined}
        {...restProps}
      >
        <div className="relative">
          <TabsList
            ref={tabsRef}
            className={cn(
              tabVariants[variant],
              fullWidth && 'w-full',
              'relative'
            )}
          >
            {tabs.map((tab) => (
              <TabsTrigger
                key={tab.value}
                value={tab.value}
                disabled={tab.disabled}
                className={cn(
                  tabTriggerVariants[variant],
                  tabSizes[size],
                  fullWidth && 'flex-1',
                  'relative transition-all duration-200'
                )}
              >
                <span className="flex items-center gap-2">
                  {tab.icon}
                  {tab.label}
                  {tab.badge !== undefined && (
                    <span className="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-primary/10 text-primary">
                      {tab.badge}
                    </span>
                  )}
                </span>
              </TabsTrigger>
            ))}

            {/* Animated underline indicator */}
            {variant === 'underline' && (
              <div
                className="absolute bottom-0 h-0.5 bg-primary transition-all duration-200"
                style={indicatorStyle}
              />
            )}
          </TabsList>
        </div>

        {children}
      </Tabs>
    );
  }
);

// Re-export TabsContent for convenience
export { TabsContent };

// ============================================================================
// Enhanced Pagination
// ============================================================================

export interface EnhancedPaginationProps extends React.HTMLAttributes<HTMLElement> {
  /** Current page (1-indexed) */
  currentPage: number;
  /** Total pages */
  totalPages: number;
  /** On page change */
  onPageChange: (page: number) => void;
  /** Show first/last buttons */
  showFirstLast?: boolean;
  /** Sibling count (pages shown on each side of current) */
  siblingCount?: number;
  /** Size */
  size?: 'sm' | 'md' | 'lg';
  /** Show page info */
  showPageInfo?: boolean;
}

function generatePaginationRange(
  currentPage: number,
  totalPages: number,
  siblingCount: number
): (number | 'ellipsis')[] {
  const totalNumbers = siblingCount * 2 + 3; // siblings + current + first + last
  const totalBlocks = totalNumbers + 2; // + 2 ellipsis

  if (totalPages <= totalBlocks) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }

  const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
  const rightSiblingIndex = Math.min(currentPage + siblingCount, totalPages);

  const showLeftEllipsis = leftSiblingIndex > 2;
  const showRightEllipsis = rightSiblingIndex < totalPages - 1;

  if (!showLeftEllipsis && showRightEllipsis) {
    const leftItemCount = 3 + 2 * siblingCount;
    const leftRange = Array.from({ length: leftItemCount }, (_, i) => i + 1);
    return [...leftRange, 'ellipsis', totalPages];
  }

  if (showLeftEllipsis && !showRightEllipsis) {
    const rightItemCount = 3 + 2 * siblingCount;
    const rightRange = Array.from(
      { length: rightItemCount },
      (_, i) => totalPages - rightItemCount + i + 1
    );
    return [1, 'ellipsis', ...rightRange];
  }

  const middleRange = Array.from(
    { length: rightSiblingIndex - leftSiblingIndex + 1 },
    (_, i) => leftSiblingIndex + i
  );
  return [1, 'ellipsis', ...middleRange, 'ellipsis', totalPages];
}

const paginationSizes = {
  sm: 'h-8 w-8 text-xs',
  md: 'h-9 w-9 text-sm',
  lg: 'h-10 w-10 text-base',
};

export const EnhancedPagination = forwardRef<HTMLElement, EnhancedPaginationProps>(
  function EnhancedPagination(
    {
      className,
      currentPage,
      totalPages,
      onPageChange,
      showFirstLast = true,
      siblingCount = 1,
      size = 'md',
      showPageInfo = false,
      ...props
    },
    ref
  ) {
    const pages = generatePaginationRange(currentPage, totalPages, siblingCount);
    const sizeClass = paginationSizes[size];

    return (
      <nav
        ref={ref}
        className={cn('flex items-center gap-1', className)}
        aria-label="Pagination"
        {...props}
      >
        {showFirstLast && (
          <Button
            variant="outline"
            size="icon"
            className={sizeClass}
            onClick={() => onPageChange(1)}
            disabled={currentPage === 1}
            aria-label="First page"
          >
            <ChevronLeft className="h-4 w-4" />
            <ChevronLeft className="h-4 w-4 -ml-2" />
          </Button>
        )}

        <Button
          variant="outline"
          size="icon"
          className={sizeClass}
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          aria-label="Previous page"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        {pages.map((page, index) =>
          page === 'ellipsis' ? (
            <span
              key={`ellipsis-${index}`}
              className={cn('flex items-center justify-center', sizeClass)}
            >
              ...
            </span>
          ) : (
            <Button
              key={page}
              variant={currentPage === page ? 'default' : 'outline'}
              size="icon"
              className={cn(
                sizeClass,
                'transition-all duration-200',
                currentPage === page && 'scale-110'
              )}
              onClick={() => onPageChange(page)}
              aria-label={`Page ${page}`}
              aria-current={currentPage === page ? 'page' : undefined}
            >
              {page}
            </Button>
          )
        )}

        <Button
          variant="outline"
          size="icon"
          className={sizeClass}
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          aria-label="Next page"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>

        {showFirstLast && (
          <Button
            variant="outline"
            size="icon"
            className={sizeClass}
            onClick={() => onPageChange(totalPages)}
            disabled={currentPage === totalPages}
            aria-label="Last page"
          >
            <ChevronRight className="h-4 w-4" />
            <ChevronRight className="h-4 w-4 -ml-2" />
          </Button>
        )}

        {showPageInfo && (
          <span className="ml-4 text-sm text-muted-foreground">
            Page {currentPage} of {totalPages}
          </span>
        )}
      </nav>
    );
  }
);

// ============================================================================
// Step Indicator
// ============================================================================

export interface StepIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Steps */
  steps: StepItem[];
  /** Current step (0-indexed) */
  currentStep: number;
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Size */
  size?: 'sm' | 'md' | 'lg';
  /** On step click */
  onStepClick?: (step: number) => void;
  /** Allow clicking completed steps */
  clickableCompleted?: boolean;
}

const stepSizes = {
  sm: {
    circle: 'h-6 w-6 text-xs',
    connector: 'h-0.5',
    label: 'text-xs',
    description: 'text-xs',
  },
  md: {
    circle: 'h-8 w-8 text-sm',
    connector: 'h-0.5',
    label: 'text-sm',
    description: 'text-xs',
  },
  lg: {
    circle: 'h-10 w-10 text-base',
    connector: 'h-1',
    label: 'text-base',
    description: 'text-sm',
  },
};

export const StepIndicator = forwardRef<HTMLDivElement, StepIndicatorProps>(
  function StepIndicator(
    {
      className,
      steps,
      currentStep,
      orientation = 'horizontal',
      size = 'md',
      onStepClick,
      clickableCompleted = true,
      ...props
    },
    ref
  ) {
    const sizeStyles = stepSizes[size];
    const isHorizontal = orientation === 'horizontal';

    const getStepStatus = (index: number) => {
      if (index < currentStep) return 'completed';
      if (index === currentStep) return 'current';
      return 'upcoming';
    };

    const handleStepClick = (index: number) => {
      if (!onStepClick) return;
      const status = getStepStatus(index);
      if (status === 'completed' && clickableCompleted) {
        onStepClick(index);
      }
    };

    return (
      <div
        ref={ref}
        className={cn(
          'flex',
          isHorizontal ? 'flex-row items-start' : 'flex-col',
          className
        )}
        {...props}
      >
        {steps.map((step, index) => {
          const status = getStepStatus(index);
          const isLast = index === steps.length - 1;
          const isClickable = onStepClick && (status === 'completed' && clickableCompleted);

          return (
            <div
              key={index}
              className={cn(
                'flex',
                isHorizontal ? 'flex-col items-center flex-1' : 'flex-row items-start'
              )}
            >
              <div
                className={cn(
                  'flex',
                  isHorizontal ? 'flex-row items-center w-full' : 'flex-col items-center'
                )}
              >
                {/* Step circle */}
                <button
                  type="button"
                  onClick={() => handleStepClick(index)}
                  disabled={!isClickable}
                  className={cn(
                    'flex items-center justify-center rounded-full border-2 transition-all duration-200',
                    sizeStyles.circle,
                    status === 'completed' && 'bg-primary border-primary text-primary-foreground',
                    status === 'current' && 'border-primary text-primary',
                    status === 'upcoming' && 'border-muted-foreground/30 text-muted-foreground',
                    isClickable && 'cursor-pointer hover:scale-110'
                  )}
                >
                  {status === 'completed' ? (
                    <Check className="h-4 w-4" />
                  ) : step.icon ? (
                    step.icon
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </button>

                {/* Connector */}
                {!isLast && (
                  <div
                    className={cn(
                      'flex-1',
                      isHorizontal ? 'mx-2' : 'my-2',
                      isHorizontal ? sizeStyles.connector : 'w-0.5 h-8',
                      status === 'completed' ? 'bg-primary' : 'bg-muted-foreground/30',
                      'transition-colors duration-200'
                    )}
                  />
                )}
              </div>

              {/* Label and description */}
              <div
                className={cn(
                  isHorizontal ? 'mt-2 text-center' : 'ml-3',
                  'min-w-0'
                )}
              >
                <p
                  className={cn(
                    'font-medium',
                    sizeStyles.label,
                    status === 'current' && 'text-primary',
                    status === 'upcoming' && 'text-muted-foreground'
                  )}
                >
                  {step.label}
                </p>
                {step.description && (
                  <p
                    className={cn(
                      'text-muted-foreground mt-0.5',
                      sizeStyles.description
                    )}
                  >
                    {step.description}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  }
);

// ============================================================================
// Command Palette
// ============================================================================

export interface CommandPaletteProps {
  /** Is open */
  open: boolean;
  /** On open change */
  onOpenChange: (open: boolean) => void;
  /** Commands */
  commands: CommandItem[];
  /** Placeholder */
  placeholder?: string;
  /** Empty message */
  emptyMessage?: string;
}

export function CommandPalette({
  open,
  onOpenChange,
  commands,
  placeholder = 'Type a command or search...',
  emptyMessage = 'No results found.',
}: CommandPaletteProps) {
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Filter commands
  const filteredCommands = commands.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(search.toLowerCase()) ||
      cmd.description?.toLowerCase().includes(search.toLowerCase())
  );

  // Group commands
  const groupedCommands = filteredCommands.reduce<Record<string, CommandItem[]>>(
    (acc, cmd) => {
      const group = cmd.group || 'Commands';
      if (!acc[group]) acc[group] = [];
      acc[group].push(cmd);
      return acc;
    },
    {}
  );

  // Reset on open
  useEffect(() => {
    if (open) {
      setSearch('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  // Keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < filteredCommands.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : filteredCommands.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action();
          onOpenChange(false);
        }
        break;
      case 'Escape':
        e.preventDefault();
        onOpenChange(false);
        break;
    }
  };

  // Global keyboard shortcut
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        onOpenChange(!open);
      }
    };

    document.addEventListener('keydown', handleGlobalKeyDown);
    return () => document.removeEventListener('keydown', handleGlobalKeyDown);
  }, [open, onOpenChange]);

  let flatIndex = 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="p-0 gap-0 max-w-lg overflow-hidden">
        <div className="flex items-center border-b px-3">
          <Search className="h-4 w-4 text-muted-foreground mr-2" />
          <Input
            ref={inputRef}
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="border-0 focus-visible:ring-0 h-12"
          />
          <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
            <span className="text-xs">⌘</span>K
          </kbd>
        </div>

        <ScrollArea className="max-h-[300px]">
          {filteredCommands.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              {emptyMessage}
            </div>
          ) : (
            <div className="p-2">
              {Object.entries(groupedCommands).map(([group, items]) => (
                <div key={group}>
                  <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                    {group}
                  </div>
                  {items.map((cmd) => {
                    const currentIndex = flatIndex++;
                    const isSelected = currentIndex === selectedIndex;

                    return (
                      <button
                        key={cmd.id}
                        onClick={() => {
                          cmd.action();
                          onOpenChange(false);
                        }}
                        onMouseEnter={() => setSelectedIndex(currentIndex)}
                        className={cn(
                          'w-full flex items-center gap-3 px-2 py-2 rounded-md text-sm',
                          'transition-colors duration-100',
                          isSelected
                            ? 'bg-accent text-accent-foreground'
                            : 'hover:bg-accent/50'
                        )}
                      >
                        {cmd.icon && (
                          <span className="flex-shrink-0 text-muted-foreground">
                            {cmd.icon}
                          </span>
                        )}
                        <div className="flex-1 text-left">
                          <div className="font-medium">{cmd.label}</div>
                          {cmd.description && (
                            <div className="text-xs text-muted-foreground">
                              {cmd.description}
                            </div>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
                            {cmd.shortcut}
                          </kbd>
                        )}
                        {isSelected && (
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </ScrollArea>

        <div className="flex items-center justify-between border-t px-3 py-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <kbd className="px-1.5 py-0.5 rounded border bg-muted">↑↓</kbd>
            <span>Navigate</span>
          </div>
          <div className="flex items-center gap-2">
            <kbd className="px-1.5 py-0.5 rounded border bg-muted">↵</kbd>
            <span>Select</span>
          </div>
          <div className="flex items-center gap-2">
            <kbd className="px-1.5 py-0.5 rounded border bg-muted">Esc</kbd>
            <span>Close</span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================================
// Command Palette Trigger Hook
// ============================================================================

export function useCommandPalette() {
  const [open, setOpen] = useState(false);

  const toggle = useCallback(() => setOpen((prev) => !prev), []);
  const openPalette = useCallback(() => setOpen(true), []);
  const closePalette = useCallback(() => setOpen(false), []);

  return {
    open,
    setOpen,
    toggle,
    openPalette,
    closePalette,
  };
}
