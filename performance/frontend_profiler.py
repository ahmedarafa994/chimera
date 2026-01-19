"""
Frontend Performance Monitoring with Core Web Vitals
Monitors React/Next.js application performance, rendering metrics, and user experience
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

# Browser automation for performance testing
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from profiling_config import MetricType, config


@dataclass
class CoreWebVitals:
    """Core Web Vitals metrics"""

    lcp: float  # Largest Contentful Paint
    fid: float  # First Input Delay
    cls: float  # Cumulative Layout Shift
    fcp: float  # First Contentful Paint
    ttfb: float  # Time to First Byte
    timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Comprehensive frontend performance metrics"""

    url: str
    timestamp: datetime
    core_web_vitals: CoreWebVitals
    load_times: dict[str, float]
    resource_metrics: dict[str, Any]
    javascript_errors: list[str]
    network_requests: list[dict[str, Any]]
    user_timing: dict[str, float]
    memory_usage: dict[str, float] | None


@dataclass
class UserJourneyMetrics:
    """User journey performance metrics"""

    journey_name: str
    steps: list[dict[str, Any]]
    total_duration: float
    success: bool
    error_messages: list[str]
    performance_budget: dict[str, float]
    budget_violations: list[str]


@dataclass
class FrontendPerformanceReport:
    """Comprehensive frontend performance report"""

    report_id: str
    timestamp: datetime
    url: str
    performance_metrics: list[PerformanceMetrics]
    user_journeys: list[UserJourneyMetrics]
    performance_score: float
    recommendations: list[str]
    budget_analysis: dict[str, Any]


class FrontendProfiler:
    """Frontend performance profiler using browser automation"""

    def __init__(self):
        self.metrics_history: list[PerformanceMetrics] = []
        self.journey_metrics: list[UserJourneyMetrics] = []

    def measure_core_web_vitals(
        self, url: str, use_playwright: bool = True
    ) -> CoreWebVitals | None:
        """Measure Core Web Vitals using browser automation"""
        if use_playwright and PLAYWRIGHT_AVAILABLE:
            return self._measure_with_playwright(url)
        elif SELENIUM_AVAILABLE:
            return self._measure_with_selenium(url)
        else:
            print("No browser automation library available")
            return None

    def _measure_with_playwright(self, url: str) -> CoreWebVitals | None:
        """Measure performance using Playwright"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
                )

                context = browser.new_context()
                page = context.new_page()

                # Enable performance tracking
                page.on("console", lambda msg: print(f"Console: {msg.text}"))

                start_time = time.time()
                page.goto(url, wait_until="networkidle")
                load_time = time.time() - start_time

                # Execute JavaScript to get Web Vitals
                web_vitals_script = """
                () => {
                    return new Promise((resolve) => {
                        const metrics = {
                            lcp: 0,
                            fid: 0,
                            cls: 0,
                            fcp: 0,
                            ttfb: 0
                        };

                        // Get performance navigation timing
                        const perfData = performance.getEntriesByType('navigation')[0];
                        if (perfData) {
                            metrics.ttfb = perfData.responseStart - perfData.requestStart;
                        }

                        // Get paint metrics
                        const paintMetrics = performance.getEntriesByType('paint');
                        paintMetrics.forEach(metric => {
                            if (metric.name === 'first-contentful-paint') {
                                metrics.fcp = metric.startTime;
                            }
                        });

                        // Use Web Vitals library if available
                        if (window.webVitals) {
                            window.webVitals.getLCP((metric) => {
                                metrics.lcp = metric.value;
                            });

                            window.webVitals.getFID((metric) => {
                                metrics.fid = metric.value;
                            });

                            window.webVitals.getCLS((metric) => {
                                metrics.cls = metric.value;
                            });
                        }

                        // Fallback measurements
                        setTimeout(() => {
                            if (metrics.lcp === 0) {
                                // Estimate LCP as largest image or text block load time
                                const images = document.querySelectorAll('img');
                                const texts = document.querySelectorAll('h1, h2, p');
                                metrics.lcp = Math.max(metrics.fcp, load_time * 1000);
                            }

                            resolve(metrics);
                        }, 1000);
                    });
                }
                """

                metrics = page.evaluate(web_vitals_script)

                browser.close()

                return CoreWebVitals(
                    lcp=metrics.get("lcp", load_time * 1000),
                    fid=metrics.get("fid", 0),
                    cls=metrics.get("cls", 0),
                    fcp=metrics.get("fcp", load_time * 1000),
                    ttfb=metrics.get("ttfb", 0),
                    timestamp=datetime.now(UTC),
                )

        except Exception as e:
            print(f"Error measuring with Playwright: {e}")
            return None

    def _measure_with_selenium(self, url: str) -> CoreWebVitals | None:
        """Measure performance using Selenium WebDriver"""
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")

            driver = webdriver.Chrome(options=options)

            start_time = time.time()
            driver.get(url)

            # Wait for page to load
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            load_time = time.time() - start_time

            # Get performance metrics
            perf_script = """
            var perfData = performance.getEntriesByType('navigation')[0];
            var paintMetrics = performance.getEntriesByType('paint');

            var result = {
                ttfb: perfData ? perfData.responseStart - perfData.requestStart : 0,
                fcp: 0,
                lcp: 0,
                loadTime: perfData ? perfData.loadEventEnd - perfData.fetchStart : 0
            };

            paintMetrics.forEach(function(metric) {
                if (metric.name === 'first-contentful-paint') {
                    result.fcp = metric.startTime;
                }
            });

            // Estimate LCP as load time if not available
            result.lcp = result.fcp > 0 ? Math.max(result.fcp, result.loadTime) : result.loadTime;

            return result;
            """

            metrics = driver.execute_script(perf_script)

            driver.quit()

            return CoreWebVitals(
                lcp=metrics.get("lcp", load_time * 1000),
                fid=0,  # FID cannot be measured programmatically
                cls=0,  # CLS requires more complex measurement
                fcp=metrics.get("fcp", load_time * 1000),
                ttfb=metrics.get("ttfb", 0),
                timestamp=datetime.now(UTC),
            )

        except Exception as e:
            print(f"Error measuring with Selenium: {e}")
            return None

    def comprehensive_page_analysis(self, url: str) -> PerformanceMetrics | None:
        """Perform comprehensive page performance analysis"""
        if not PLAYWRIGHT_AVAILABLE:
            print("Playwright required for comprehensive analysis")
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()

                # Track network requests
                network_requests = []
                javascript_errors = []

                page.on(
                    "request",
                    lambda request: network_requests.append(
                        {
                            "url": request.url,
                            "method": request.method,
                            "resource_type": request.resource_type,
                            "timestamp": time.time(),
                        }
                    ),
                )

                page.on(
                    "console",
                    lambda msg: javascript_errors.append(msg.text) if msg.type == "error" else None,
                )

                # Navigate and measure
                start_time = time.time()
                page.goto(url, wait_until="networkidle")
                end_time = time.time()

                # Get Core Web Vitals
                core_vitals = self._get_web_vitals_from_page(page)

                # Get resource metrics
                resource_metrics = page.evaluate(
                    """
                () => {
                    const resources = performance.getEntriesByType('resource');
                    const resourceSummary = {
                        total_requests: resources.length,
                        total_transfer_size: 0,
                        resource_types: {}
                    };

                    resources.forEach(resource => {
                        if (resource.transferSize) {
                            resourceSummary.total_transfer_size += resource.transferSize;
                        }

                        const type = resource.initiatorType || 'other';
                        resourceSummary.resource_types[type] =
                            (resourceSummary.resource_types[type] || 0) + 1;
                    });

                    return resourceSummary;
                }
                """
                )

                # Get user timing metrics
                user_timing = page.evaluate(
                    """
                () => {
                    const userTimings = {};
                    const marks = performance.getEntriesByType('mark');
                    const measures = performance.getEntriesByType('measure');

                    marks.forEach(mark => {
                        userTimings[mark.name] = mark.startTime;
                    });

                    measures.forEach(measure => {
                        userTimings[measure.name] = measure.duration;
                    });

                    return userTimings;
                }
                """
                )

                # Get memory usage (if available)
                memory_usage = None
                import contextlib

                with contextlib.suppress(Exception):
                    memory_usage = page.evaluate(
                        """
                    () => {
                        if (performance.memory) {
                            return {
                                used: performance.memory.usedJSHeapSize,
                                total: performance.memory.totalJSHeapSize,
                                limit: performance.memory.jsHeapSizeLimit
                            };
                        }
                        return null;
                    }
                    """
                    )

                browser.close()

                load_times = {
                    "total_load_time": (end_time - start_time) * 1000,
                    "dom_content_loaded": 0,  # Would need more complex tracking
                    "first_paint": core_vitals.fcp if core_vitals else 0,
                    "largest_contentful_paint": core_vitals.lcp if core_vitals else 0,
                }

                metrics = PerformanceMetrics(
                    url=url,
                    timestamp=datetime.now(UTC),
                    core_web_vitals=core_vitals or CoreWebVitals(0, 0, 0, 0, 0, datetime.now(UTC)),
                    load_times=load_times,
                    resource_metrics=resource_metrics,
                    javascript_errors=javascript_errors,
                    network_requests=network_requests,
                    user_timing=user_timing,
                    memory_usage=memory_usage,
                )

                self.metrics_history.append(metrics)
                return metrics

        except Exception as e:
            print(f"Error in comprehensive page analysis: {e}")
            return None

    def _get_web_vitals_from_page(self, page) -> CoreWebVitals | None:
        """Extract Web Vitals from current page"""
        try:
            vitals = page.evaluate(
                """
            () => {
                return new Promise((resolve) => {
                    const metrics = { lcp: 0, fid: 0, cls: 0, fcp: 0, ttfb: 0 };

                    const perfData = performance.getEntriesByType('navigation')[0];
                    if (perfData) {
                        metrics.ttfb = perfData.responseStart - perfData.requestStart;
                    }

                    const paintMetrics = performance.getEntriesByType('paint');
                    paintMetrics.forEach(metric => {
                        if (metric.name === 'first-contentful-paint') {
                            metrics.fcp = metric.startTime;
                        }
                    });

                    // Estimate LCP
                    metrics.lcp = Math.max(metrics.fcp, 1000);

                    setTimeout(() => resolve(metrics), 100);
                });
            }
            """
            )

            return CoreWebVitals(
                lcp=vitals["lcp"],
                fid=vitals["fid"],
                cls=vitals["cls"],
                fcp=vitals["fcp"],
                ttfb=vitals["ttfb"],
                timestamp=datetime.now(UTC),
            )
        except Exception:
            return None

    def test_user_journey(self, journey_config: dict[str, Any]) -> UserJourneyMetrics:
        """Test a complete user journey"""
        journey_name = journey_config["name"]
        steps = journey_config.get("steps", [])
        performance_budget = journey_config.get("performance_budget", {})

        journey_steps = []
        error_messages = []
        total_start_time = time.time()

        if not PLAYWRIGHT_AVAILABLE:
            return UserJourneyMetrics(
                journey_name=journey_name,
                steps=[],
                total_duration=0,
                success=False,
                error_messages=["Playwright not available"],
                performance_budget=performance_budget,
                budget_violations=[],
            )

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()

                for i, step in enumerate(steps):
                    step_start_time = time.time()
                    step_success = True
                    step_error = None

                    try:
                        if step["action"] == "navigate":
                            page.goto(step["url"], wait_until="networkidle")

                        elif step["action"] == "click":
                            page.click(step["selector"])
                            page.wait_for_load_state("networkidle")

                        elif step["action"] == "type":
                            page.fill(step["selector"], step["text"])

                        elif step["action"] == "wait":
                            page.wait_for_timeout(step["duration"])

                        elif step["action"] == "wait_for_element":
                            page.wait_for_selector(
                                step["selector"], timeout=step.get("timeout", 30000)
                            )

                    except Exception as e:
                        step_success = False
                        step_error = str(e)
                        error_messages.append(f"Step {i+1} ({step.get('name', 'unnamed')}): {e}")

                    step_duration = (time.time() - step_start_time) * 1000

                    journey_steps.append(
                        {
                            "step_name": step.get("name", f"step_{i+1}"),
                            "action": step["action"],
                            "duration_ms": step_duration,
                            "success": step_success,
                            "error": step_error,
                        }
                    )

                browser.close()

        except Exception as e:
            error_messages.append(f"Journey execution error: {e}")

        total_duration = (time.time() - total_start_time) * 1000
        success = len(error_messages) == 0

        # Check performance budget violations
        budget_violations = []
        if (
            performance_budget.get("max_total_time")
            and total_duration > performance_budget["max_total_time"]
        ):
            budget_violations.append(
                f"Total journey time {total_duration:.0f}ms exceeds budget {performance_budget['max_total_time']}ms"
            )

        for step in journey_steps:
            if (
                performance_budget.get("max_step_time")
                and step["duration_ms"] > performance_budget["max_step_time"]
            ):
                budget_violations.append(
                    f"Step '{step['step_name']}' time {step['duration_ms']:.0f}ms exceeds budget {performance_budget['max_step_time']}ms"
                )

        journey_metrics = UserJourneyMetrics(
            journey_name=journey_name,
            steps=journey_steps,
            total_duration=total_duration,
            success=success,
            error_messages=error_messages,
            performance_budget=performance_budget,
            budget_violations=budget_violations,
        )

        self.journey_metrics.append(journey_metrics)
        return journey_metrics

    def generate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Generate overall performance score (0-100)"""
        score = 100

        # Core Web Vitals scoring
        if metrics.core_web_vitals.lcp > 4000:  # Poor LCP (>4s)
            score -= 30
        elif metrics.core_web_vitals.lcp > 2500:  # Needs improvement (>2.5s)
            score -= 15

        if metrics.core_web_vitals.fcp > 3000:  # Poor FCP (>3s)
            score -= 20
        elif metrics.core_web_vitals.fcp > 1800:  # Needs improvement (>1.8s)
            score -= 10

        if metrics.core_web_vitals.cls > 0.25:  # Poor CLS (>0.25)
            score -= 20
        elif metrics.core_web_vitals.cls > 0.1:  # Needs improvement (>0.1)
            score -= 10

        # Load time scoring
        if metrics.load_times["total_load_time"] > 5000:  # >5s
            score -= 15
        elif metrics.load_times["total_load_time"] > 3000:  # >3s
            score -= 8

        # Resource efficiency
        if metrics.resource_metrics.get("total_requests", 0) > 100:
            score -= 10

        # JavaScript errors
        if len(metrics.javascript_errors) > 5:
            score -= 10
        elif len(metrics.javascript_errors) > 0:
            score -= 5

        return max(0, score)

    def generate_recommendations(self, metrics: PerformanceMetrics) -> list[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Core Web Vitals recommendations
        if metrics.core_web_vitals.lcp > 2500:
            recommendations.extend(
                [
                    "Optimize Largest Contentful Paint by optimizing images and critical resources",
                    "Implement lazy loading for below-the-fold images",
                    "Use a Content Delivery Network (CDN) for faster resource delivery",
                ]
            )

        if metrics.core_web_vitals.fcp > 1800:
            recommendations.extend(
                [
                    "Reduce First Contentful Paint by optimizing critical rendering path",
                    "Minimize render-blocking resources (CSS/JS)",
                    "Implement critical CSS inlining",
                ]
            )

        if metrics.core_web_vitals.cls > 0.1:
            recommendations.extend(
                [
                    "Reduce Cumulative Layout Shift by setting dimensions for images and ads",
                    "Avoid inserting content above existing content",
                    "Use CSS aspect-ratio for responsive images",
                ]
            )

        # Resource optimization
        if metrics.resource_metrics.get("total_requests", 0) > 100:
            recommendations.extend(
                [
                    "Reduce number of HTTP requests by bundling resources",
                    "Implement resource minification and compression",
                    "Use HTTP/2 server push for critical resources",
                ]
            )

        # JavaScript optimization
        if len(metrics.javascript_errors) > 0:
            recommendations.extend(
                [
                    "Fix JavaScript errors to improve user experience",
                    "Implement proper error handling and logging",
                    "Use source maps for better debugging",
                ]
            )

        # Memory optimization
        if metrics.memory_usage and metrics.memory_usage.get("used", 0) > 50 * 1024 * 1024:  # >50MB
            recommendations.extend(
                [
                    "Optimize JavaScript memory usage",
                    "Implement object pooling for frequently created objects",
                    "Use weak references where appropriate",
                ]
            )

        return recommendations

    def save_performance_report(self, url: str, metrics_list: list[PerformanceMetrics]) -> str:
        """Save comprehensive performance report"""
        report_id = f"frontend_report_{int(time.time())}"
        timestamp = datetime.now(UTC)

        # Calculate overall performance score
        avg_score = sum(self.generate_performance_score(m) for m in metrics_list) / len(
            metrics_list
        )

        # Aggregate recommendations
        all_recommendations = []
        for metrics in metrics_list:
            all_recommendations.extend(self.generate_recommendations(metrics))

        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)

        # Performance budget analysis
        budget_analysis = {
            "lcp_budget": 2500,
            "fcp_budget": 1800,
            "cls_budget": 0.1,
            "violations": [],
        }

        for metrics in metrics_list:
            if metrics.core_web_vitals.lcp > budget_analysis["lcp_budget"]:
                budget_analysis["violations"].append(
                    f"LCP violation: {metrics.core_web_vitals.lcp:.0f}ms > {budget_analysis['lcp_budget']}ms"
                )

        report = FrontendPerformanceReport(
            report_id=report_id,
            timestamp=timestamp,
            url=url,
            performance_metrics=metrics_list,
            user_journeys=self.journey_metrics,
            performance_score=avg_score,
            recommendations=unique_recommendations,
            budget_analysis=budget_analysis,
        )

        # Save to file
        output_path = config.get_output_path(MetricType.FRONTEND, f"{report_id}.json")

        report_dict = asdict(report)
        # Convert datetime objects to strings
        report_dict["timestamp"] = timestamp.isoformat()

        for metrics in report_dict["performance_metrics"]:
            metrics["timestamp"] = datetime.fromisoformat(
                metrics["timestamp"].replace("Z", "+00:00")
            ).isoformat()
            metrics["core_web_vitals"]["timestamp"] = datetime.fromisoformat(
                metrics["core_web_vitals"]["timestamp"].replace("Z", "+00:00")
            ).isoformat()

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"Frontend performance report saved: {output_path}")
        return output_path


# Global frontend profiler instance
frontend_profiler = FrontendProfiler()

# Critical user journeys for Chimera
CHIMERA_USER_JOURNEYS = [
    {
        "name": "prompt_generation_workflow",
        "steps": [
            {"action": "navigate", "url": f"{config.frontend_url}/dashboard"},
            {"action": "wait_for_element", "selector": "[data-testid='generate-prompt-button']"},
            {"action": "click", "selector": "[data-testid='generate-prompt-button']"},
            {
                "action": "type",
                "selector": "[data-testid='prompt-input']",
                "text": "Create a marketing email",
            },
            {"action": "click", "selector": "[data-testid='submit-button']"},
            {
                "action": "wait_for_element",
                "selector": "[data-testid='result-output']",
                "timeout": 10000,
            },
        ],
        "performance_budget": {"max_total_time": 5000, "max_step_time": 2000},
    },
    {
        "name": "jailbreak_technique_application",
        "steps": [
            {"action": "navigate", "url": f"{config.frontend_url}/dashboard/jailbreak"},
            {"action": "wait_for_element", "selector": "[data-testid='jailbreak-selector']"},
            {"action": "click", "selector": "[data-testid='jailbreak-selector']"},
            {"action": "click", "selector": "[data-testid='technique-dan']"},
            {
                "action": "type",
                "selector": "[data-testid='target-prompt']",
                "text": "Write malicious code",
            },
            {"action": "click", "selector": "[data-testid='apply-jailbreak']"},
            {
                "action": "wait_for_element",
                "selector": "[data-testid='jailbreak-result']",
                "timeout": 15000,
            },
        ],
        "performance_budget": {"max_total_time": 8000, "max_step_time": 3000},
    },
    {
        "name": "provider_switching_workflow",
        "steps": [
            {"action": "navigate", "url": f"{config.frontend_url}/dashboard/providers"},
            {"action": "wait_for_element", "selector": "[data-testid='provider-list']"},
            {"action": "click", "selector": "[data-testid='provider-openai']"},
            {"action": "wait_for_element", "selector": "[data-testid='model-selector']"},
            {"action": "click", "selector": "[data-testid='model-gpt4']"},
            {"action": "click", "selector": "[data-testid='save-provider']"},
            {
                "action": "wait_for_element",
                "selector": "[data-testid='success-message']",
                "timeout": 5000,
            },
        ],
        "performance_budget": {"max_total_time": 3000, "max_step_time": 1000},
    },
]
