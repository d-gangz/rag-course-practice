<!--
Document Type: Learning Notes
Purpose: Comprehensive guide to understanding semaphores in Python asyncio with progressive examples
Context: Created while learning async/await patterns for concurrent API calls in RAG course
Key Topics: Semaphores, concurrency control, async/await, rate limiting, resource management
Target Use: Reference guide for understanding and applying semaphores in concurrent programming
-->

# Understanding Semaphores in Python AsyncIO

## What is a Semaphore?

A **semaphore** is a concurrency control mechanism that limits how many tasks can access a resource simultaneously.

Think of it like a **parking lot with limited spaces**:
- The parking lot has 5 spaces (capacity)
- When a car arrives and there's space, it parks
- When all 5 spaces are full, new cars must wait
- When a car leaves, a waiting car can park

## Why Do We Need Semaphores?

**Problem**: Making 1000 API calls at once can:
- Hit rate limits (e.g., OpenAI allows 500 requests/minute)
- Overwhelm servers
- Cause memory issues
- Result in errors and retries

**Solution**: Use a semaphore to limit concurrent operations to a safe number (e.g., 10 at a time).

---

## 🎯 Critical Concept: What Semaphores Actually Control

**IMPORTANT**: This is the most common misunderstanding about semaphores.

### ❌ Common Misconception
"If I have `Semaphore(3)` and create 6 workers, only 3 workers run at a time."

### ✅ Reality
**All 6 workers run simultaneously!** The semaphore only controls access to **specific code sections** inside the workers.

### How It Actually Works

```python
async def worker(name: str, sem: Semaphore):
    # ========== ZONE 1: UNRESTRICTED ==========
    # ALL 6 workers execute this section simultaneously
    print(f"{name}: Preparing...")
    prepare_data()

    # ========== ZONE 2: SEMAPHORE GATE ==========
    async with sem:  # ← Only 3 workers can be HERE at once
        # The expensive/protected section
        print(f"{name}: Inside gate, doing expensive work")
        await expensive_api_call()

    # ========== ZONE 3: UNRESTRICTED ==========
    # ALL 6 workers execute this section simultaneously
    print(f"{name}: Cleaning up...")
    cleanup()

async def main():
    sem = Semaphore(3)  # Gate capacity = 3

    # Coroutine loop - creates all 6 workers
    coros = [worker(f"W{i}", sem) for i in range(6)]

    # All 6 workers start immediately!
    await asyncio.gather(*coros)
```

### Visual Timeline

```
Time 0s: All 6 workers start and run Zone 1 simultaneously
         ↓
         W0, W1, W2 enter the gate (async with sem:) - Gate full (3/3)
         W3, W4, W5 reach the gate but WAIT (gate is full)

Time 2s: W0, W1, W2 finish and exit the gate
         ↓
         W3, W4, W5 now enter the gate (3/3)
         W0, W1, W2 continue to Zone 3 simultaneously

Time 4s: W3, W4, W5 finish and exit the gate
         ↓
         All 6 workers continue to Zone 3 simultaneously
```

### Key Takeaways

1. **`async with sem:` is a GATE**, not a worker on/off switch
2. **All workers run simultaneously**, but queue at the gate
3. **Only N workers** (N = semaphore limit) can execute the code inside `async with sem:` at once
4. **Place the gate around expensive operations** (API calls, heavy I/O, resource-limited tasks)

### Real-World Analogy: Restaurant Kitchen

```python
async def delivery_driver(name: str, sem: Semaphore):
    # 🚗 Unrestricted: All 6 drivers drive to restaurant simultaneously
    await drive_to_restaurant()

    # 🍳 GATE: Only 3 orders can be cooked at once (kitchen capacity)
    async with sem:
        await kitchen_cook_order()  # Takes 10 minutes

    # 🚗 Unrestricted: All 6 drivers deliver simultaneously
    await deliver_to_customer()
```

All 6 drivers are working the whole time, but only 3 can have food cooked simultaneously because the kitchen capacity is limited.

---

## Example 1: Basic Semaphore (Beginner)

Let's see how the semaphore gate works with a clear demonstration.

```python
import asyncio
from asyncio import Semaphore

async def worker(name: str, sem: Semaphore):
    """A worker that needs to access a limited resource."""
    # ========== ALL 6 workers execute this line ==========
    print(f"{name}: Started! Approaching the gate...")

    # ========== GATE: Only 3 workers can pass at once ==========
    async with sem:  # Workers wait here if gate is full
        print(f"{name}: Inside the gate! Working on expensive task...")
        await asyncio.sleep(2)  # Simulate expensive work
        print(f"{name}: Finished expensive work, exiting gate.")

    # ========== ALL 6 workers execute this line ==========
    print(f"{name}: Past the gate, all done!")

async def main():
    sem = Semaphore(3)  # Gate capacity = 3 workers at once

    # Coroutine loop: Create 6 workers
    coros = [worker(f"Worker-{i}", sem) for i in range(6)]

    # All 6 workers start immediately!
    await asyncio.gather(*coros)

# Run it
asyncio.run(main())
```

**Output:**
```
Worker-0: Started! Approaching the gate...
Worker-1: Started! Approaching the gate...
Worker-2: Started! Approaching the gate...
Worker-3: Started! Approaching the gate...
Worker-4: Started! Approaching the gate...
Worker-5: Started! Approaching the gate...
# ↑ ALL 6 workers started simultaneously!

Worker-0: Inside the gate! Working on expensive task...
Worker-1: Inside the gate! Working on expensive task...
Worker-2: Inside the gate! Working on expensive task...
# ↑ First 3 workers pass through the gate (3/3 slots filled)
# Workers 3, 4, 5 are WAITING at the gate

# ... 2 seconds pass ...

Worker-0: Finished expensive work, exiting gate.
Worker-0: Past the gate, all done!
Worker-1: Finished expensive work, exiting gate.
Worker-1: Past the gate, all done!
Worker-2: Finished expensive work, exiting gate.
Worker-2: Past the gate, all done!

Worker-3: Inside the gate! Working on expensive task...
Worker-4: Inside the gate! Working on expensive task...
Worker-5: Inside the gate! Working on expensive task...
# ↑ Next 3 workers now pass through the gate

# ... 2 seconds pass ...

Worker-3: Finished expensive work, exiting gate.
Worker-3: Past the gate, all done!
Worker-4: Finished expensive work, exiting gate.
Worker-4: Past the gate, all done!
Worker-5: Finished expensive work, exiting gate.
Worker-5: Past the gate, all done!
```

**Key Points:**
- **All 6 workers start immediately** and run the code before `async with sem:`
- **`async with sem:` is the GATE** - only 3 workers can be inside at once
- Workers that reach a full gate **wait** at the `async with sem:` line
- When a worker exits the gate, a waiting worker can enter
- Automatic release when exiting `async with` block
- **The semaphore ONLY limits the code inside `async with sem:`**, not the entire worker function

---

## Example 2: API Calls with Semaphore (Intermediate)

Now let's simulate API calls with rate limiting.

```python
import asyncio
from asyncio import Semaphore
import time

async def call_api(request_id: int, sem: Semaphore):
    """Simulate an API call with rate limiting."""
    async with sem:  # Only N concurrent calls allowed
        print(f"[{time.strftime('%H:%M:%S')}] Request {request_id}: Starting API call")
        await asyncio.sleep(1)  # Simulate API response time
        print(f"[{time.strftime('%H:%M:%S')}] Request {request_id}: Completed")
        return f"Response for request {request_id}"

async def main():
    sem = Semaphore(5)  # Max 5 concurrent API calls

    # Create 20 API calls (coroutines)
    coros = [call_api(i, sem) for i in range(20)]

    # Run all (but only 5 will run at a time due to semaphore)
    start = time.time()
    results = await asyncio.gather(*coros)
    elapsed = time.time() - start

    print(f"\n✓ Completed {len(results)} API calls in {elapsed:.2f} seconds")
    print(f"Average time per call: {elapsed/len(results):.2f}s")
    print(f"With semaphore=5, we run in batches: 20 calls / 5 concurrent = 4 batches × 1s = ~4s")

asyncio.run(main())
```

**Output:**
```
[14:30:00] Request 0: Starting API call
[14:30:00] Request 1: Starting API call
[14:30:00] Request 2: Starting API call
[14:30:00] Request 3: Starting API call
[14:30:00] Request 4: Starting API call
# ... 1 second passes ...
[14:30:01] Request 0: Completed
[14:30:01] Request 1: Completed
[14:30:01] Request 2: Completed
[14:30:01] Request 3: Completed
[14:30:01] Request 4: Completed
[14:30:01] Request 5: Starting API call  # Next batch starts
[14:30:01] Request 6: Starting API call
...
✓ Completed 20 API calls in 4.01 seconds
```

**Key Insight:**
- Without semaphore: All 20 calls start at once → potential rate limit errors
- With semaphore=5: Calls run in controlled batches of 5 → safe and predictable

---

## Example 3: Real OpenAI API Calls (Advanced)

Let's use a real-world example with OpenAI API and error handling.

```python
import asyncio
from asyncio import Semaphore
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

client = AsyncOpenAI()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate_completion(prompt: str, sem: Semaphore) -> str:
    """Generate completion with rate limiting and retry logic."""
    async with sem:  # Rate limiting
        print(f"🔵 Processing: '{prompt[:30]}...'")

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            result = response.choices[0].message.content
            print(f"✅ Completed: '{prompt[:30]}...'")
            return result

        except Exception as e:
            print(f"❌ Error for '{prompt[:30]}...': {e}")
            raise  # Tenacity will retry

async def main():
    # Limit to 10 concurrent API calls (OpenAI rate limit)
    sem = Semaphore(10)

    # Create 50 prompts
    prompts = [f"Write a haiku about {topic}"
               for topic in ["ocean", "mountain", "forest", "desert", "city"] * 10]

    # Generate all completions (coroutines)
    coros = [generate_completion(prompt, sem) for prompt in prompts]
    results = await asyncio.gather(*coros, return_exceptions=True)

    # Count successes
    successes = [r for r in results if isinstance(r, str)]
    failures = [r for r in results if isinstance(r, Exception)]

    print(f"\n📊 Results:")
    print(f"   ✓ Successful: {len(successes)}")
    print(f"   ✗ Failed: {len(failures)}")

asyncio.run(main())
```

**Why This Works:**
1. **Semaphore(10)** = max 10 concurrent API calls (respects rate limits)
2. **@retry decorator** = automatic retry on failures
3. **async with sem** = acquires slot, waits if all 10 slots are full
4. **asyncio.gather** = runs all tasks efficiently

---

## Example 4: Comparing With vs Without Semaphore (Eye-Opening)

Let's see the difference in behavior.

```python
import asyncio
import time
from asyncio import Semaphore

async def api_call(request_id: int, sem: Semaphore = None):
    """Simulate an API call that might fail if too many run at once."""
    if sem:
        async with sem:
            return await _do_api_call(request_id)
    else:
        return await _do_api_call(request_id)

async def _do_api_call(request_id: int):
    """Simulate API that fails if >10 concurrent requests."""
    # Count how many tasks are running right now
    current_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

    if current_tasks > 10:
        raise Exception(f"Rate limit exceeded! ({current_tasks} concurrent)")

    await asyncio.sleep(0.5)  # Simulate API delay
    return f"Success: {request_id}"

async def test_without_semaphore():
    """Run 50 API calls without rate limiting."""
    print("\n🚫 WITHOUT SEMAPHORE:")
    coros = [api_call(i) for i in range(50)]
    results = await asyncio.gather(*coros, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]
    print(f"   Failed requests: {len(failures)}")

async def test_with_semaphore():
    """Run 50 API calls with rate limiting."""
    print("\n✅ WITH SEMAPHORE(10):")
    sem = Semaphore(10)
    coros = [api_call(i, sem) for i in range(50)]
    results = await asyncio.gather(*coros, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]
    print(f"   Failed requests: {len(failures)}")

async def main():
    await test_without_semaphore()
    await test_with_semaphore()

asyncio.run(main())
```

**Output:**
```
🚫 WITHOUT SEMAPHORE:
   Failed requests: 35  # Most requests fail due to rate limiting!

✅ WITH SEMAPHORE(10):
   Failed requests: 0   # All succeed because we stay under limit
```

---

## Example 5: Dynamic Semaphore Adjustment (Expert)

Sometimes you want to adjust the semaphore limit dynamically based on success/failure rates.

```python
import asyncio
from asyncio import Semaphore

class AdaptiveSemaphore:
    """Semaphore that adjusts based on success rate."""

    def __init__(self, initial_limit: int = 10):
        self.limit = initial_limit
        self.sem = Semaphore(initial_limit)
        self.successes = 0
        self.failures = 0

    async def __aenter__(self):
        await self.sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.sem.release()

        if exc_type is None:
            self.successes += 1
            # If doing well, increase concurrency
            if self.successes % 10 == 0 and self.limit < 50:
                self.limit += 5
                print(f"📈 Increasing limit to {self.limit}")
                self._adjust_semaphore()
        else:
            self.failures += 1
            # If failing, decrease concurrency
            if self.failures % 3 == 0 and self.limit > 5:
                self.limit -= 5
                print(f"📉 Decreasing limit to {self.limit}")
                self._adjust_semaphore()

    def _adjust_semaphore(self):
        """Recreate semaphore with new limit."""
        self.sem = Semaphore(self.limit)

async def api_call_with_adaptive(request_id: int, sem: AdaptiveSemaphore):
    """API call with adaptive rate limiting."""
    async with sem:
        await asyncio.sleep(0.1)
        # Randomly fail 10% of the time
        if request_id % 10 == 0:
            raise Exception("Random failure")
        return f"Success: {request_id}"

async def main():
    sem = AdaptiveSemaphore(initial_limit=10)

    coros = [api_call_with_adaptive(i, sem) for i in range(100)]
    results = await asyncio.gather(*coros, return_exceptions=True)

    print(f"\n📊 Final Stats:")
    print(f"   Successes: {sem.successes}")
    print(f"   Failures: {sem.failures}")
    print(f"   Final limit: {sem.limit}")

asyncio.run(main())
```

---

## Key Takeaways

### When to Use Semaphores

✅ **Use semaphores when:**
- Making many API calls (rate limiting)
- Accessing limited resources (database connections, file handles)
- Controlling memory usage (limit concurrent heavy operations)
- Preventing server overload

❌ **Don't use semaphores when:**
- Tasks are CPU-bound (use ProcessPoolExecutor instead)
- You need strict ordering (use Queue instead)
- Resource is truly unlimited

### Semaphore vs Other Patterns

| **Pattern** | **Use Case** | **Example** |
|-------------|--------------|-------------|
| **Semaphore** | Limit concurrent access | Max 10 API calls at once |
| **Lock** | Only 1 task at a time | Writing to a shared file |
| **Queue** | Task ordering matters | Job processing pipeline |
| **ThreadPoolExecutor** | CPU-bound work | Image processing |

### Best Practices

1. **Choose the right limit:**
   - Too low → slow (underutilized)
   - Too high → rate limit errors
   - Start with API provider's limit ÷ 2

2. **Always use `async with`:**
   ```python
   # ✅ Good - automatic release
   async with sem:
       await do_work()

   # ❌ Bad - manual release (easy to forget)
   await sem.acquire()
   await do_work()
   sem.release()
   ```

3. **Combine with retry logic:**
   ```python
   @retry(stop=stop_after_attempt(3))
   async def api_call(sem):
       async with sem:
           return await client.call()
   ```

4. **Monitor and adjust:**
   - Log semaphore wait times
   - Track success/failure rates
   - Adjust limits based on performance

---

## 🎯 CRITICAL DECISION: Semaphore vs ThreadPoolExecutor

This is the #1 confusion point! Here's the definitive guide:

### Understanding I/O-bound vs CPU-bound

**I/O-bound (Input/Output) = WAITING for external resources**
- Your code sends a request, then **waits** for a response
- During the wait, your **CPU is IDLE** (doing nothing)
- Examples: API calls, database queries, file operations, network requests
- **This is 95% of your work if you're doing LLM/API/database tasks!**

**CPU-bound = ACTIVELY COMPUTING**
- Your code is **actively calculating** something
- Your **CPU is BUSY** working (not waiting)
- Examples: Image processing, video encoding, complex math, data parsing
- **You probably won't encounter this much in typical LLM/API work**

---

### The Golden Rule (For Your Daily Work)

**Step 1:** Most of your tasks are I/O-bound (API calls, database queries)

**Step 2:** Check if the SDK supports async:
```
Does the SDK/library have `async` or `await` in the docs?
│
├─ YES → ✅ Use asyncio + Semaphore
│   Examples: OpenAI (AsyncOpenAI), Anthropic (AsyncAnthropic),
│            httpx, aiofiles, asyncpg, Braintrust (await Eval)
│
└─ NO → ✅ Use ThreadPoolExecutor
    Examples: requests library, old Cohere SDK, sync database drivers
```

**Step 3:** CPU-bound? (Rare for your work)
```
Are you doing heavy computation (not API calls)?
│
└─ YES → Use multiprocessing.Pool
    Examples: Image processing, video encoding, complex algorithms

    ⚠️  You'll RARELY need this for LLM/API/database work!
```

---

### Real-World Reality Check

**For typical LLM/RAG/API work, you'll only use:**
1. **asyncio + Semaphore** (90% of the time) - OpenAI, Anthropic, modern APIs
2. **ThreadPoolExecutor** (10% of the time) - Old sync libraries, simple scripts

**You'll almost NEVER use:**
3. **multiprocessing.Pool** - Unless you're processing images, videos, or heavy computation

**The simplified decision for your daily work:**
```
I'm making API calls or querying databases...
│
└─ Check the docs: Does it have `await` or `AsyncClient`?
    ├─ YES → asyncio + Semaphore ✅
    └─ NO → ThreadPoolExecutor ✅
```

---

### 💡 How They Work Differently (The Key Insight!)

**The Core Problem: You want to run 100 API calls efficiently**

#### **Async (Non-blocking) - The Problem & Solution:**

**What happens WITHOUT Semaphore:**
```python
# ❌ All 100 async functions fire AT ONCE (no automatic queue!)
async def call_api(i):
    response = await api.get(f"/item/{i}")
    return response

# All 100 START IMMEDIATELY - no queuing!
tasks = [call_api(i) for i in range(100)]
await asyncio.gather(*tasks)
# ← 100 requests hit API simultaneously! Rate limit error! ❌
```

- **Async = non-blocking**, so all 100 calls start **immediately**
- No automatic queuing - they all fire at once
- Result: API overload, rate limit errors

**What happens WITH Semaphore:**
```python
# ✅ Semaphore limits to 10 at a time
sem = Semaphore(10)

async def call_api(i, sem):
    async with sem:  # ← GATE: Only 10 can be here at once
        response = await api.get(f"/item/{i}")
    return response

# All 100 START, but only 10 pass through semaphore gate
tasks = [call_api(i, sem) for i in range(100)]
await asyncio.gather(*tasks)  # ✅ Only 10 concurrent calls at a time
```

- All 100 coroutines start immediately
- Semaphore acts as a **gate** - only 10 can execute API call
- Other 90 **wait** at `async with sem:` line
- Result: Controlled concurrency, no rate limit errors

---

#### **Sync (Blocking) - The Problem & Solution:**

**What happens WITHOUT ThreadPoolExecutor:**
```python
# ❌ All 100 sync functions run ONE AT A TIME (sequential queue!)
def call_api_sync(i):
    response = api.get(f"/item/{i}")  # Blocking!
    return response

results = []
for i in range(100):
    results.append(call_api_sync(i))  # ← One finishes, then next starts
# First → Second → Third → ... → 100th (very slow!)
```

- **Sync = blocking**, so functions run **one at a time** (queued)
- First finishes → Second starts → Third starts... (sequential)
- No concurrency at all
- Result: 100 calls take 100× longer

**What happens WITH ThreadPoolExecutor:**
```python
# ✅ ThreadPoolExecutor creates 10 parallel threads
def call_api_sync(i):
    response = api.get(f"/item/{i}")  # Blocking
    return response

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(call_api_sync, range(100)))
# ✅ 10 calls at a time (10 threads working in parallel)
```

- Creates 10 separate threads
- Each thread handles sync functions sequentially
- 10 threads = 10 functions running at same time
- Result: 10× faster than sequential

---

#### **Both Achieve "10 at a Time" - Different Mechanisms:**

| Aspect | Async + Semaphore | ThreadPoolExecutor |
|--------|-------------------|-------------------|
| **Without control** | All 100 start at once ❌ | All 100 queue one-by-one ❌ |
| **With control** | Semaphore gates 10 at a time ✅ | 10 threads run 10 at a time ✅ |
| **How it works** | 100 coroutines exist, semaphore limits access | 10 threads exist, naturally limited |
| **Memory** | 1KB × 100 = 100KB | 8MB × 10 = 80MB |
| **Max concurrent** | 100-1000+ | 10-50 |

---

#### **Visual Timeline: 100 Tasks, Limit to 10 Concurrent**

**Async + Semaphore:**
```
Time 0s:  All 100 coroutines START immediately
          ↓
          Coroutines 1-10:  [Inside gate, calling API] ← 10 active
          Coroutines 11-100: [Waiting at gate] ← 90 waiting

Time 2s:  Coroutine 1 done → Coroutine 11 enters gate
          Coroutines 2-10, 11: [Inside gate] ← 10 active
          Coroutines 12-100: [Waiting] ← 89 waiting

...continues until all 100 done
```

**ThreadPoolExecutor:**
```
Time 0s:  Only 10 threads exist (not 100!)
          ↓
          Thread 1:  [Task 1] → [Task 11] → [Task 21] → ...
          Thread 2:  [Task 2] → [Task 12] → [Task 22] → ...
          ...
          Thread 10: [Task 10] → [Task 20] → [Task 30] → ...

Each thread processes tasks sequentially
Only 10 tasks active at any time
```

---

#### **Why Asyncio + Semaphore is Better for I/O:**
- **1KB per task** (lightweight coroutines) vs **8MB per thread** (heavy)
- Can handle **100+ concurrent** easily vs **10-20 threads** max
- **Faster switching** (event loop) vs **slower** (OS thread context switching)
- **No GIL contention** (one thread) vs **GIL switching** (multiple threads)

---

#### **When ThreadPoolExecutor Still Works:**
- Sync SDK with I/O operations (requests, old libraries)
- I/O operations **release the Python GIL**, so threads can wait concurrently
- Not as efficient as asyncio, but works fine for smaller scale (10-20 concurrent)

---

### 🎯 The Simple Mental Model:

**Async without Semaphore:**
- 🏃💨💨💨 All 100 people RUN to the door at once → **STAMPEDE!** ❌

**Async with Semaphore:**
- 🏃💨💨💨 All 100 people START running, but door only lets 10 through at a time ✅
- (100 people running, 10 inside, 90 waiting at door)

**Sync without ThreadPoolExecutor:**
- 🚶...🚶...🚶 People line up single-file → one at a time through door ❌
- (Very slow, 100 in a sequential queue)

**Sync with ThreadPoolExecutor:**
- 👥👥👥 Create 10 doors (threads) → 10 people through at once ✅
- (10 threads, each processing tasks sequentially)

---

### Decision Tree with Code Examples

#### Scenario 1: Async SDK Available (BEST - Use Asyncio + Semaphore)

```python
# ✅ OpenAI SDK supports async
from openai import AsyncOpenAI
from asyncio import Semaphore

client = AsyncOpenAI()

async def call_openai(prompt, sem):
    async with sem:
        response = await client.chat.completions.create(...)  # ← `await` works!
        return response.choices[0].message.content

async def main():
    sem = Semaphore(10)
    coros = [call_openai(p, sem) for p in prompts]
    results = await asyncio.gather(*coros)

# Why this is BEST:
# ✅ Lightweight (1KB per coroutine vs 8MB per thread)
# ✅ Can handle 100+ concurrent operations
# ✅ Fast context switching (event loop)
# ✅ No Python GIL issues
# ✅ Built-in rate limiting with semaphore
```

#### Scenario 2: No Async SDK (Use ThreadPoolExecutor)

```python
# ❌ Old Cohere SDK (no async support)
import cohere
from concurrent.futures import ThreadPoolExecutor

co = cohere.Client(api_key)

def call_cohere(prompt):
    response = co.generate(prompt=prompt)  # ← No `await`! Blocking!
    return response.generations[0].text

# ✅ Use ThreadPoolExecutor since SDK is sync
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(call_cohere, prompts))

# Why ThreadPoolExecutor here:
# ✅ Works with sync I/O operations
# ✅ Each thread can wait independently (I/O releases GIL)
# ⚠️  Limited to ~10-20 threads (memory overhead)
# ⚠️  8MB per thread (can't scale to 100+ like async)
```

#### Scenario 3: CPU-Bound Work (Use multiprocessing)

```python
# CPU-bound: Image processing
from multiprocessing import Pool
import numpy as np

def process_image(image_path):
    img = load_image(image_path)
    # Heavy computation: resize, filter, transform
    processed = apply_filters(img)  # ← CPU working hard!
    return processed

# ✅ Use multiprocessing for CPU work
with Pool(processes=8) as pool:  # Use all CPU cores
    results = pool.map(process_image, image_paths)

# Why multiprocessing:
# ✅ True parallelism (separate Python processes)
# ✅ No GIL limitation (each process has own GIL)
# ✅ Utilizes all CPU cores
# ❌ High memory overhead (full Python interpreter per process)
# ❌ NOT for I/O-bound tasks
```

---

### When Each Approach Fails

#### ❌ Using Asyncio with Sync SDK (DOESN'T WORK!)

```python
# ❌ WRONG - Trying to use async with sync SDK
import asyncio
from asyncio import Semaphore

def sync_api_call(prompt):  # ← Sync function, no `await`
    return sync_client.generate(prompt)

async def worker(prompt, sem):
    async with sem:
        result = sync_api_call(prompt)  # ← BLOCKS event loop!
        # Other coroutines CAN'T run while this executes
        return result

# This gives you NO benefit! All tasks run one-at-a-time.
# The event loop is blocked during sync_api_call()
```

#### ❌ Using ThreadPoolExecutor for CPU Work (SLOW!)

```python
# ❌ WRONG - Threads for CPU-bound work
from concurrent.futures import ThreadPoolExecutor

def calculate_heavy(data):
    # CPU-intensive calculation
    result = complex_math(data)  # ← CPU actively computing
    return result

# ❌ Threads won't help here due to Python GIL
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(calculate_heavy, dataset))

# Why this is SLOW:
# - Python GIL allows only ONE thread to run Python code at a time
# - All 8 threads compete for the GIL
# - No true parallelism for CPU work
# - Use multiprocessing.Pool instead!
```

---

### The Memory & Scalability Comparison

| Approach | Memory per Task | Max Concurrent | Use Case |
|----------|----------------|----------------|----------|
| **asyncio + Semaphore** | ~1 KB | 100-1000+ | I/O-bound with async SDK |
| **ThreadPoolExecutor** | ~8 MB | 10-50 | I/O-bound with sync SDK |
| **multiprocessing.Pool** | ~50 MB | 1-16 (CPU cores) | CPU-bound work |

**Example:**
- **100 API calls with asyncio**: ~100 KB memory ✅
- **100 API calls with threads**: ~800 MB memory ❌
- **100 CPU tasks with multiprocessing**: ~5 GB memory (impractical!)

---

### Real-World Decision Examples

#### Example 1: OpenAI API Calls
```python
# Task: Generate 500 completions
# I/O-bound? YES (waiting for API response)
# Async SDK? YES (AsyncOpenAI exists)
# Decision: ✅ asyncio + Semaphore

from openai import AsyncOpenAI
from asyncio import Semaphore

async def generate(prompt, sem):
    async with sem:
        return await client.chat.completions.create(...)

# Can handle 50+ concurrent easily
```

#### Example 2: Web Scraping (Sync Library)
```python
# Task: Scrape 200 websites
# I/O-bound? YES (waiting for HTTP response)
# Async SDK? NO (using requests library - sync only)
# Decision: ✅ ThreadPoolExecutor

import requests
from concurrent.futures import ThreadPoolExecutor

def scrape(url):
    return requests.get(url).text  # Sync, but I/O releases GIL

with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(scrape, urls)
```

#### Example 3: Image Processing
```python
# Task: Resize 1000 images
# I/O-bound? NO (CPU computing pixel transformations)
# Decision: ✅ multiprocessing.Pool

from multiprocessing import Pool
from PIL import Image

def resize(image_path):
    img = Image.open(image_path)
    return img.resize((800, 600))  # CPU-intensive

with Pool(processes=8) as pool:  # Use all CPU cores
    pool.map(resize, image_paths)
```

#### Example 4: Database Queries (Async Support)
```python
# Task: Query 300 records from PostgreSQL
# I/O-bound? YES (waiting for database)
# Async SDK? YES (asyncpg library)
# Decision: ✅ asyncio + Semaphore

import asyncpg
from asyncio import Semaphore

async def query_db(user_id, sem):
    async with sem:
        conn = await asyncpg.connect(...)
        return await conn.fetch("SELECT * FROM users WHERE id=$1", user_id)

# Efficient concurrent DB queries
```

#### Example 5: LanceDB Local Search (Sync + CPU-bound)
```python
# Task: Search 500 queries in local LanceDB
# I/O-bound? NO (local disk + vector similarity computation)
# CPU-bound? YES (vector math: dot products, sorting)
# Async SDK? NO (LanceDB Python is sync)
# Decision: ✅ multiprocessing.Pool

from multiprocessing import Pool

def search_db(query):
    results = table.search(query).limit(10)  # CPU: vector similarity
    return results.to_list()

with Pool(processes=8) as pool:
    all_results = pool.map(search_db, queries)
```

---

### Workaround: Async with Sync SDK (Advanced)

If you MUST use asyncio with a sync SDK, wrap it in a thread pool:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Sync SDK
def sync_api_call(prompt):
    return sync_client.generate(prompt)

# Wrap in executor
async def async_wrapper(prompt, sem):
    async with sem:
        loop = asyncio.get_event_loop()
        # Run sync function in thread pool
        result = await loop.run_in_executor(None, sync_api_call, prompt)
        return result

async def main():
    sem = Semaphore(10)
    coros = [async_wrapper(p, sem) for p in prompts]
    return await asyncio.gather(*coros)

# This works, but ThreadPoolExecutor is simpler for this case
```

---

### 🎯 The Simple Decision Flow

```
START: What am I doing?
│
├─ Making API calls / Database queries / Network requests
│   └─ Does the library have `async` or `await` in the docs?
│       ├─ YES → asyncio + Semaphore ✅
│       └─ NO → ThreadPoolExecutor ✅
│
├─ Heavy computation / Image processing / Data transformation
│   └─ multiprocessing.Pool ✅
│
└─ Mixed (API call + computation)
    └─ Use asyncio for I/O, then process results with multiprocessing
```

---

### Summary Table

| Your Task | Question to Ask | Answer | Use This |
|-----------|----------------|--------|----------|
| OpenAI API calls | Has async SDK? | ✅ Yes | asyncio + Semaphore |
| Old Cohere API | Has async SDK? | ❌ No | ThreadPoolExecutor |
| Web scraping (requests) | Has async SDK? | ❌ No | ThreadPoolExecutor |
| Web scraping (httpx) | Has async SDK? | ✅ Yes | asyncio + Semaphore |
| LanceDB local search | I/O or CPU? | CPU (vector math) | multiprocessing.Pool |
| LanceDB cloud search | Has async SDK? | Check docs first | Depends on SDK |
| Image resizing | I/O or CPU? | CPU | multiprocessing.Pool |
| File reading | Has async SDK? | ✅ Yes (aiofiles) | asyncio + Semaphore |
| Braintrust Eval | Has async SDK? | ✅ Yes (await Eval) | asyncio + Semaphore |

---

### Key Takeaways

1. **ALWAYS check if the SDK has async support FIRST** (look for `await`, `async def`, `AsyncClient` in docs)

2. **I/O-bound + async SDK = asyncio + Semaphore** (best performance, lowest memory)

3. **I/O-bound + sync SDK = ThreadPoolExecutor** (threads release GIL during I/O wait)

4. **CPU-bound = multiprocessing.Pool** (true parallelism, uses all cores)

5. **When in doubt:**
   - Try asyncio first if SDK has async
   - Fall back to ThreadPoolExecutor if SDK is sync but I/O-bound
   - Use multiprocessing only for heavy computation

6. **Python GIL rule:**
   - Threads DON'T help CPU work (GIL blocks parallelism)
   - Threads DO help I/O work (GIL released during I/O wait)
   - Async BEST for I/O if SDK supports it (no thread overhead)

---

## Real-World Example: Your RAG Course Code

From your `01_syn_ques_eg.py`:

```python
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
async def generate_questions(chunk: Chunk, sem: Semaphore) -> ChunkEval:
    async with sem:  # Rate limiting (max 10 concurrent)
        coro = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[...],
            response_model=Question,
        )
        resp = await asyncio.wait_for(coro, timeout=30)
        return ChunkEval(...)

async def main():
    sem = Semaphore(10)  # Max 10 concurrent OpenAI calls

    # Create 100 tasks
    coros = [generate_questions(chunk, sem) for chunk in dataset]

    # Run all efficiently (only 10 at a time)
    results = await tqdm_asyncio.gather(*coros)
```

**Why this is perfect:**
- OpenAI rate limit: ~500 requests/min
- Semaphore(10) = max 10 concurrent
- Stays well under rate limit
- Efficient memory usage
- Automatic retry on failures

---

## Practice Exercise

Try converting the ThreadPoolExecutor code from your homework to use async + semaphore!

**Challenge:** Take the `synthetic_queries.py` file and convert:
- `ThreadPoolExecutor` → `asyncio + Semaphore`
- `completion()` → `acompletion()`
- `time.sleep()` → `await asyncio.sleep()`
- `as_completed()` → `tqdm_asyncio.gather()`

You'll see significant performance improvements! 🚀

---

## 📚 Quick Reference: Semaphore Pattern

### The Standard Pattern

```python
# 1. Parent function with coroutine loop
async def main():
    sem = Semaphore(N)  # Define gate capacity

    # Coroutine loop - creates all workers
    coros = [worker(data, sem) for data in dataset]

    # Run all workers (choose one):
    results = await asyncio.gather(*coros)  # No progress bar
    # OR
    results = await tqdm_asyncio.gather(*coros)  # With progress bar ✨

# 2. Worker function with semaphore gate
async def worker(data, sem: Semaphore):
    # Unrestricted code (all workers can execute)
    prepare_data(data)

    # GATE: Only N workers can be here at once
    async with sem:
        result = await expensive_operation(data)

    # Unrestricted code (all workers can execute)
    cleanup(result)
    return result
```

### Progress Bar with tqdm_asyncio

For long-running operations with many tasks, use `tqdm_asyncio.gather()` to see progress:

```python
import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm_asyncio  # Add this import

async def main():
    sem = Semaphore(10)
    coros = [generate_question(chunk, sem) for chunk in dataset]

    # Shows a nice progress bar as tasks complete!
    results = await tqdm_asyncio.gather(*coros)
    # Output: 100%|██████████| 100/100 [00:45<00:00, 2.21it/s]
```

**When to use:**
- ✅ **Use `tqdm_asyncio.gather()`** when:
  - Running 50+ tasks
  - Operations take more than a few seconds
  - You want to see progress (development/debugging)
  - You want to estimate time remaining

- ✅ **Use `asyncio.gather()`** when:
  - Running < 10 quick tasks
  - Production scripts (no human watching)
  - You don't need visual feedback

**Note:** They're functionally identical - `tqdm_asyncio.gather()` just adds a progress bar on top of `asyncio.gather()`.

**Example from your code:**
```python
# week-1/01_syn_ques_eg.py line 105
questions: list[ChunkEval] = await tqdm_asyncio.gather(*coros)
# Shows progress for generating 100+ questions
```

### Mental Model Checklist

✅ **Remember:**
1. `async with sem:` is a **GATE** for specific code sections
2. **All workers run simultaneously** - they just queue at the gate
3. **Only N workers** can execute code inside `async with sem:` at once
4. Place the gate **around expensive operations** (API calls, heavy I/O)
5. Workers **wait** at `async with sem:` if the gate is full
6. When a worker exits the gate, a waiting worker enters

❌ **Don't think:**
- "Semaphore limits how many workers run" ← Wrong! All workers run.
- "Semaphore controls the entire function" ← Wrong! Only controls the `async with sem:` block.

### Coroutines ≈ Promises (JavaScript/TypeScript)

| **Python** | **JavaScript/TypeScript** |
|-----------|---------------------------|
| `coro = async_func()` | `promise = asyncFunc()` |
| `await coro` | `await promise` |
| `asyncio.gather(*coros)` | `Promise.all(promises)` |
| `async with sem:` | No direct equivalent (unique to Python) |

---

## 🎓 Summary

Semaphores are essential for controlling concurrent access to limited resources in async Python:

- **Use case**: Rate limiting API calls, managing database connections, controlling resource usage
- **Key concept**: Semaphore is a gate for specific code sections, not a worker limiter
- **Best for**: I/O-bound operations (API calls, file I/O, network requests)
- **Not for**: CPU-bound operations (use `ProcessPoolExecutor` instead)

Master semaphores and you'll write efficient, rate-limit-friendly concurrent code! 🚀
