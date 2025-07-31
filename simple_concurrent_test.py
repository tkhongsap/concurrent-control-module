import os
import asyncio
import time
import json
import random
from datetime import datetime
import dotenv
from openai import AzureOpenAI
from typing import Dict, List
from tabulate import tabulate

# Load environment variables
dotenv.load_dotenv()

class SimpleConcurrentTester:
    def __init__(self, total_requests: int = 200, max_concurrent: int = 20):
        self.total_requests = total_requests
        self.max_concurrent = max_concurrent
        
        # Azure OpenAI setup (reusing your existing pattern)
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-12-01-preview"
        
        # Validate required environment variables
        if not all([self.endpoint, self.deployment, self.api_key]):
            raise ValueError("Missing required environment variables. Check AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY")
        
        # Create Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Metrics tracking
        self.results: List[Dict] = []
        self.completed_requests = 0
        self.failed_requests = 0
        self.active_requests = 0
        self.start_time = None
        
        # For detailed logging
        self.detailed_logs: List[Dict] = []
        self.request_start_time = None  # When all requests are initiated
        
    async def make_single_request(self, request_id: int) -> Dict:
        """Make a single request to Azure OpenAI with retry logic"""
        request_start = time.time()
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        # Create message with current datetime and random number to make each request unique
        random_number = random.randint(1, 1000)
        user_message = f"Request {request_id} initiated at {self.request_start_time}. What is {random_number} times 2? Also give me a brief motivational quote inspired by Anthony Bourdain."
        
        # Log when request is about to wait for semaphore
        waiting_time = time.time()
        print(f"⏳ Request {request_id}: Waiting for semaphore slot... (Active: {self.active_requests}/{self.max_concurrent})")
        
        async with self.semaphore:  # This ensures max concurrent requests
            acquired_time = time.time()
            wait_duration = acquired_time - waiting_time
            self.active_requests += 1
            
            print(f"🚀 Request {request_id}: Acquired semaphore! Waited {wait_duration:.3f}s (Active: {self.active_requests}/{self.max_concurrent})")
            
            for attempt in range(max_retries + 1):
                try:
                    api_call_start = time.time()
                    print(f"📤 Request {request_id}: Sending to Azure OpenAI...")
                    print(f"   Message: {user_message}")
                    
                    response = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant. Respond briefly.",
                            },
                            {
                                "role": "user",
                                "content": user_message,
                            }
                        ],
                        max_completion_tokens=100,  # Increased for math + quote
                        temperature=0.7,
                        model=self.deployment
                    )
                    
                    api_call_end = time.time()
                    api_latency = api_call_end - api_call_start
                    request_end = time.time()
                    total_latency = request_end - request_start
                    
                    response_content = response.choices[0].message.content if response.choices else "No response"
                    tokens_used = response.usage.total_tokens if response.usage else 0
                    
                    print(f"📥 Request {request_id}: SUCCESS! API latency: {api_latency:.3f}s")
                    print(f"   Response: {response_content[:100]}{'...' if len(response_content) > 100 else ''}")
                    print(f"   Tokens used: {tokens_used}")
                    
                    # Log detailed info for table
                    detailed_log = {
                        "request_id": request_id,
                        "wait_time": f"{wait_duration:.3f}s",
                        "api_latency": f"{api_latency:.3f}s",
                        "total_latency": f"{total_latency:.3f}s",
                        "message_sent": user_message[:50] + "..." if len(user_message) > 50 else user_message,
                        "response_received": response_content[:50] + "..." if len(response_content) > 50 else response_content,
                        "tokens": tokens_used,
                        "status": "✅ SUCCESS",
                        "attempts": attempt + 1
                    }
                    self.detailed_logs.append(detailed_log)
                    
                    # Track successful request
                    result = {
                        "request_id": request_id,
                        "status": "success",
                        "latency": total_latency,
                        "api_latency": api_latency,
                        "wait_time": wait_duration,
                        "attempts": attempt + 1,
                        "timestamp": datetime.now().isoformat(),
                        "tokens_used": tokens_used,
                        "status_code": 200,
                        "message_sent": user_message,
                        "response_received": response_content
                    }
                    
                    self.completed_requests += 1
                    self.active_requests -= 1
                    print(f"✅ Request {request_id}: Completed! Releasing semaphore (Active: {self.active_requests}/{self.max_concurrent})")
                    return result
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check for rate limiting (429) or service unavailable (503)
                    is_retryable = "429" in error_str or "503" in error_str or "rate" in error_str.lower()
                    
                    if attempt < max_retries and is_retryable:
                        print(f"❌ Request {request_id} attempt {attempt + 1} failed: {error_str[:100]}...")
                        print(f"   Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Final failure
                        request_end = time.time()
                        total_latency = request_end - request_start
                        
                        print(f"💥 Request {request_id}: FAILED after {attempt + 1} attempts")
                        print(f"   Error: {error_str[:100]}...")
                        
                        # Log detailed info for table
                        detailed_log = {
                            "request_id": request_id,
                            "wait_time": f"{wait_duration:.3f}s",
                            "api_latency": "N/A",
                            "total_latency": f"{total_latency:.3f}s",
                            "message_sent": user_message[:50] + "..." if len(user_message) > 50 else user_message,
                            "response_received": f"ERROR: {error_str[:30]}...",
                            "tokens": 0,
                            "status": "❌ FAILED",
                            "attempts": attempt + 1
                        }
                        self.detailed_logs.append(detailed_log)
                        
                        result = {
                            "request_id": request_id,
                            "status": "failed",
                            "latency": total_latency,
                            "wait_time": wait_duration,
                            "attempts": attempt + 1,
                            "timestamp": datetime.now().isoformat(),
                            "tokens_used": 0,
                            "error": error_str,
                            "status_code": self._extract_status_code(error_str),
                            "message_sent": user_message,
                            "response_received": error_str
                        }
                        
                        self.failed_requests += 1
                        self.active_requests -= 1
                        print(f"❌ Request {request_id}: Failed! Releasing semaphore (Active: {self.active_requests}/{self.max_concurrent})")
                        return result
    
    def _extract_status_code(self, error_str: str) -> int:
        """Extract HTTP status code from error string"""
        if "429" in error_str:
            return 429
        elif "503" in error_str:
            return 503
        elif "500" in error_str:
            return 500
        else:
            return 0  # Unknown error
    
    async def run_load_test(self):
        """Execute the concurrent load test"""
        print(f"🚀 Starting concurrent load test...")
        print(f"📊 Total requests: {self.total_requests}")
        print(f"⚡ Max concurrent: {self.max_concurrent}")
        print(f"🎯 Target endpoint: {self.endpoint}")
        print("=" * 80)
        
        # Set the initiation time that will be sent in all messages
        self.request_start_time = datetime.now().isoformat()
        print(f"🕐 All requests initiated at: {self.request_start_time}")
        print(f"📝 Each request will send this datetime to prove they started together!")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Create all request tasks AT ONCE - this initiates all 40 requests simultaneously
        print(f"🔥 Creating all {self.total_requests} request tasks simultaneously...")
        tasks = []
        for i in range(self.total_requests):
            task = asyncio.create_task(self.make_single_request(i + 1))
            tasks.append(task)
        
        print(f"✅ All {self.total_requests} tasks created! The semaphore will now control max {self.max_concurrent} concurrent execution.")
        print("-" * 80)
        
        # Wait for all requests to complete (semaphore controls concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        self.results = [r for r in results if isinstance(r, dict)]
        
        # Generate final report with detailed table
        await self._generate_report()
    
    def _print_detailed_table(self):
        """Print detailed results in table format"""
        if not self.detailed_logs:
            print("No detailed logs to display")
            return
        
        # Sort by request_id for better readability
        sorted_logs = sorted(self.detailed_logs, key=lambda x: x["request_id"])
        
        # Prepare table data
        headers = ["ID", "Wait Time", "API Latency", "Total Latency", "Message Sent", "Response", "Tokens", "Status", "Attempts"]
        table_data = []
        
        for log in sorted_logs:
            table_data.append([
                log["request_id"],
                log["wait_time"],
                log["api_latency"],
                log["total_latency"],
                log["message_sent"],
                log["response_received"],
                log["tokens"],
                log["status"],
                log["attempts"]
            ])
        
        print("\n" + "="*120)
        print("📋 DETAILED REQUEST TABLE")
        print("="*120)
        print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[3, 8, 10, 12, 35, 35, 6, 10, 8]))
        print("="*120)
    
    async def _generate_report(self):
        """Generate and display the final test report"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        successful_results = [r for r in self.results if r["status"] == "success"]
        failed_results = [r for r in self.results if r["status"] == "failed"]
        
        if successful_results:
            latencies = [r["latency"] for r in successful_results]
            avg_latency = sum(latencies) / len(latencies)
            latencies.sort()
            p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0
        else:
            avg_latency = 0
            p95_latency = 0
        
        total_retries = sum(r["attempts"] - 1 for r in self.results)
        total_tokens = sum(r["tokens_used"] for r in successful_results)
        
        report = {
            "test_summary": {
                "total_requests": self.total_requests,
                "max_concurrent": self.max_concurrent,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / self.total_requests * 100,
                "total_duration": total_duration,
                "requests_per_second": self.total_requests / total_duration
            },
            "performance_metrics": {
                "average_latency": avg_latency,
                "p95_latency": p95_latency,
                "total_retries": total_retries,
                "total_tokens_used": total_tokens
            },
            "error_breakdown": {}
        }
        
        # Error breakdown
        for result in failed_results:
            status_code = result.get("status_code", "unknown")
            if status_code not in report["error_breakdown"]:
                report["error_breakdown"][status_code] = 0
            report["error_breakdown"][status_code] += 1
        
        print("\n" + "="*60)
        print("📈 LOAD TEST RESULTS")
        print("="*60)
        print(f"✅ Successful requests: {len(successful_results)}/{self.total_requests} ({len(successful_results)/self.total_requests*100:.1f}%)")
        print(f"❌ Failed requests: {len(failed_results)}")
        print(f"⏱️  Total duration: {total_duration:.2f} seconds")
        print(f"🚀 Requests per second: {self.total_requests/total_duration:.2f}")
        print(f"📊 Average latency: {avg_latency:.3f} seconds")
        print(f"📈 95th percentile latency: {p95_latency:.3f} seconds")
        print(f"🔄 Total retries: {total_retries}")
        print(f"🎯 Total tokens used: {total_tokens}")
        
        if report["error_breakdown"]:
            print(f"\n❌ Error breakdown:")
            for status_code, count in report["error_breakdown"].items():
                print(f"   Status {status_code}: {count} requests")
        
        # Save detailed report to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"load_test_report_{timestamp}.json"
        
        full_report = {
            **report,
            "detailed_results": self.results
        }
        
        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Print detailed table
        self._print_detailed_table()
        
        # Validate success criteria from PRD
        print(f"\n🎯 PRD SUCCESS CRITERIA CHECK:")
        print(f"   ✅ All {self.total_requests} requests completed: {'YES' if len(self.results) == self.total_requests else 'NO'}")
        print(f"   ✅ Max concurrent never exceeded {self.max_concurrent}: YES (enforced by semaphore)")
        print(f"   ✅ 100% success rate: {'YES' if len(successful_results) == self.total_requests else f'NO ({len(successful_results)/self.total_requests*100:.1f}%)'}")
        
        # Show concurrency insights
        print(f"\n🔍 CONCURRENCY INSIGHTS:")
        print(f"   📝 All requests sent same initiation time: {self.request_start_time}")
        print(f"   ⚡ Requests were processed in groups of max {self.max_concurrent}")
        print(f"   🕐 When one completes, the next waiting request immediately starts")
        print(f"   📊 See 'Wait Time' column in table above to understand queue behavior")

async def main():
    """Main entry point"""
    # You can customize these values
    total_requests = 40   # Modified for testing
    max_concurrent = 10   # Modified for testing
    
    try:
        tester = SimpleConcurrentTester(total_requests, max_concurrent)
        await tester.run_load_test()
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure your .env file contains:")
        print("AZURE_OPENAI_ENDPOINT=your_endpoint")
        print("AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment")
        print("AZURE_OPENAI_API_KEY=your_key")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())