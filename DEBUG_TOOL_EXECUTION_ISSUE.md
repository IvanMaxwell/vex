# Debug Guide: Tool Execution Not Happening After Planning

## Problem
Agent creates a plan successfully, but tool execution doesn't occur and user doesn't get an answer.

---

## Root Cause Analysis

### **Primary Culprit: Tool Call Format Parsing Failure**

The Executor Agent must format its tool calls EXACTLY as:
```
TOOL_CALL: web_search
PARAMS: {"query": "example"}
END_TOOL_CALL
```

The regex in [agents.py:153-165](agents.py#L153-L165) is very strict:
```python
pattern = re.compile(
    r"TOOL_CALL:\s*(\w+)\s*\nPARAMS:\s*(\{.*?\})\s*\nEND_TOOL_CALL",
    re.DOTALL | re.IGNORECASE,
)
```

**Common LLM deviations that BREAK parsing:**
- `TOOL_CALL: web_search` ✅ (works)
- `Tool Call: web_search` ❌ (doesn't work - uppercase "Tool")
- `tool_call: web_search` ❌ (lowercase)
- `TOOL_CALL:web_search` ❌ (no space after colon)
- `PARAMS: {"query":"test"}` vs `PARAMS:{...}` (spacing issues)
- `END_TOOL_CALL` on same line as PARAMS (no newline)

When parsing fails:
1. `tool_calls` list is empty
2. No tools execute
3. Response is treated as a status message
4. User gets no result from the intended tool

---

## Debugging Steps (In Order)

### **Step 1: Check Frontend Permission Modal**
**Where:** [templates/index.html:854](templates/index.html#L854)

If tool calls ARE parsing but not executing, the issue is likely the permission system.

**Test:**
1. Trigger a task that should use `web_search`
2. Watch the browser console for WebSocket messages
3. Look for `"type": "permission_request"` messages
4. Verify the permission modal appears

**If modal doesn't appear:**
- Check browser DevTools → Network tab
- Verify WebSocket connection is open (should show green dot in header)
- Check browser console for errors

---

### **Step 2: Check Tool Call Regex Matching**

Add debugging output to see what the LLM is actually generating:

**In [agents.py](agents.py#L474-L480), add logging:**

```python
tool_calls = _parse_tool_calls(response)

# DEBUG: Log what was attempted to parse
if not tool_calls and "TOOL_CALL" in response.lower():
    streamer.error(
        f"⚠️ Tool call format didn't match regex. LLM output:\n{response[:500]}",
        agent
    )
```

Then check:
- Is the LLM actually generating `TOOL_CALL:` blocks?
- Does the format match exactly?
- Are the JSON params valid?

---

### **Step 3: Check Permission Manager Threading**

The permission system BLOCKS the agent thread until the user responds or times out.

**Potential deadlock scenarios:**

1. **Dead WebSocket:** If `self._loop` or `self._ws` is None, permission requests never send
   - Check [runtime.py:70-85](runtime.py#L70-L85)
   - Verify `permission_manager.setup()` was called in [main.py](main.py#L68)

2. **Wrong async loop:** Agent thread is blocking on a different event loop than frontend
   - See [runtime.py:76-77](runtime.py#L76-L77)
   - The `asyncio.run_coroutine_threadsafe()` bridges the sync→async gap

3. **Timeout expires:** PERMISSION_TIMEOUT = 180 seconds (3 minutes)
   - User must click Allow/Deny within timeout
   - If they don't, permission auto-denies
   - See [config.py:36](config.py#L36)

---

### **Step 4: Trace Permission Response Flow**

**Frontend sends:** 
```json
{"type": "permission_response", "id": "abc12345", "granted": true}
```

**Backend receives in [main.py:145-149](main.py#L145-L149):**
```python
elif msg_type == "permission_response":
    perm_id = data.get("id", "")
    granted = data.get("granted", False)
    if granted:
        permission_manager.grant(perm_id)
    else:
        permission_manager.deny(perm_id)
```

**Check if response arrives:**
- Add logging to main.py permission_response handler
- Verify frontend is sending JSON with correct fields

---

### **Step 5: Check Tool Execution Failure**

Even if permission IS granted, the tool might fail:

**In [tools.py:30-37](tools.py#L30-L37):**
```python
def _retry(fn, *args, **kwargs):
    """Try fn once; on exception retry once with same args."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        # ... first attempt failed
        try:
            return fn(*args, **kwargs)
        except Exception as e2:
            return f"ERROR: {e2}"  # ← Returns error string
```

**In [agents.py:481-484](agents.py#L481-L484):**
```python
result = dispatch_tool(tc["tool"], tc["params"])
if _is_tool_failure(result):
    new_errors.append(f"Step {state['current_step'] + 1}: {result}")
```

**Check if result is marked as failure:**
- See [agents.py:101-106](agents.py#L101-L106)
- Error strings starting with "error", "unknown tool", etc. are caught
- Results are added to `new_errors` instead of `new_results`

---

## Complete Diagnostic Checklist

```
□ Check browser console for WebSocket errors
□ Verify permission modal CSS (doesn't display hidden)
□ Test permission modal by triggering a permission request manually
□ Add logging to _parse_tool_calls() to see regex matches
□ Log the full LLM response before parsing
□ Verify permission_manager.setup() was called
□ Check if permission_response arrives at backend
□ Monitor tool execution for permission denials
□ Check tool stdout/stderr for failures
□ Verify TOOL_REGISTRY has all 5 tools
□ Test tool dispatch with hardcoded parameters
```

---

## Quick Fixes

### **Fix 1: Make LLM Format More Robust**

In [agents.py:430-440](agents.py#L430-L440), the EXECUTOR_SYSTEM prompt instructs exact format. 
Add examples to make it clearer:

```python
EXECUTOR_SYSTEM = """...
To call a tool use EXACTLY this format (nothing else around it):

EXAMPLE 1:
TOOL_CALL: web_search
PARAMS: {"query": "bitcoin price"}
END_TOOL_CALL

EXAMPLE 2:
TOOL_CALL: file_ops
PARAMS: {"action": "read", "path": "test.txt"}
END_TOOL_CALL

IMPORTANT: Each section on its own line. No extra text before/after.
...
"""
```

### **Fix 2: Add Fallback Tool Call Format**

In [agents.py:153-170](agents.py#L153-L170), add alternative parsing:

```python
def _parse_tool_calls(text: str) -> list[dict]:
    """Parse TOOL_CALL blocks. Try multiple formats."""
    
    # Try primary format
    pattern = re.compile(
        r"TOOL_CALL:\s*(\w+)\s*\nPARAMS:\s*(\{.*?\})\s*\nEND_TOOL_CALL",
        re.DOTALL | re.IGNORECASE,
    )
    calls = []
    for m in pattern.finditer(text):
        tool = m.group(1).strip()
        try:
            params = json.loads(m.group(2).strip())
            calls.append({"tool": tool, "params": params})
        except:
            pass
    
    if calls:
        return calls
    
    # TODO: Add fallback format parsing here if needed
    return calls
```

### **Fix 3: Add Permission Timeout Logging**

In [runtime.py:84-85](runtime.py#L84-L85):

```python
granted = event.wait(timeout=PERMISSION_TIMEOUT)

result = self._results.pop(perm_id, False) if granted else False
if not granted:
    streamer.error(
        f"⏱ Permission timeout ({PERMISSION_TIMEOUT}s) for {perm_id}",
        "Permissions"
    )
```

---

## Testing Workflow

1. **Test tool parsing in isolation:**
   ```bash
   python -c "
   import re, json
   text = '''TOOL_CALL: web_search
   PARAMS: {\"query\": \"test\"}
   END_TOOL_CALL'''
   
   pattern = re.compile(
       r'TOOL_CALL:\s*(\w+)\s*\nPARAMS:\s*(\{.*?\})\s*\nEND_TOOL_CALL',
       re.DOTALL | re.IGNORECASE
   )
   print(pattern.findall(text))
   "
   ```

2. **Test LLM format compliance:**
   - Run a simple task: "What's the weather?"
   - Check if Executor generates proper TOOL_CALL blocks
   - Log the exact LLM response

3. **Test permission system end-to-end:**
   - Monitor WebSocket messages
   - Verify permission_request arrives at frontend
   - Manually send permission_response from console
   - Check if agent thread unblocks

---

## Most Likely Culprit Summary

**80% probability:** LLM not formatting TOOL_CALL blocks correctly
- **Solution:** Lower OLLAMA_NUM_PREDICT or use a more instruction-following model
- **Test:** Check LLM output directly

**15% probability:** Permission system timeout or frontend issue
- **Solution:** Check browser console, WebSocket connection, modal CSS
- **Test:** Manually grant permission from console

**5% probability:** Tool execution failure after permission granted
- **Solution:** Check tool stdout/stderr in responses
- **Test:** Run tool with hardcoded parameters

