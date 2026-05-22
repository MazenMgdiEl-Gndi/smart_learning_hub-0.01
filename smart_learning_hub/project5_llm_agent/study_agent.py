"""
PROJECT 4 — LLM-Based Study Agent
Reads a student's study log and returns:
  - Progress summary
  - Weakness identification
  - Next-session recommendation

Uses the Anthropic API (claude-haiku — fast & cheap).
Set environment variable: ANTHROPIC_API_KEY=your_key_here

Run: python study_agent.py
"""

import os
import json

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("⚠️  anthropic package not installed. Run: pip install anthropic")
    print("   Falling back to mock mode for demonstration.\n")


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a friendly and concise AI Study Coach for students.

Given a student's study log for the day, you must respond in valid JSON with:
{
  "summary": "2-sentence summary of what the student accomplished today",
  "strengths": ["list of 1-3 things the student did well"],
  "weaknesses": ["list of 1-3 areas where the student struggled"],
  "next_session_plan": {
    "focus_topic": "the single most important topic to review next",
    "recommended_activities": ["2-3 specific actions to take"],
    "estimated_time_minutes": 45
  },
  "motivation_message": "one short, genuine encouraging sentence"
}

Be specific. Never say vague things like 'keep up the good work'. Always tie advice to the actual log.
Output ONLY the JSON object, no markdown, no extra text.
""".strip()


# ── Mock fallback (no API key) ─────────────────────────────────────────────────
MOCK_RESPONSE = {
    "summary": "The student spent 2 hours on Python basics and struggled with for-loops and list comprehensions. They completed the variables and functions sections successfully.",
    "strengths": ["Completed variables and functions chapters", "Maintained focus for 2 full hours", "Attempted loop exercises multiple times"],
    "weaknesses": ["For-loop logic especially with nested loops", "List comprehension syntax is not yet comfortable"],
    "next_session_plan": {
        "focus_topic": "For-loops and list comprehensions in Python",
        "recommended_activities": [
            "Watch a 15-minute video on Python loops (e.g., Corey Schafer on YouTube)",
            "Solve 5 loop exercises on HackerRank or LeetCode Easy",
            "Rewrite previous assignments using list comprehensions"
        ],
        "estimated_time_minutes": 60
    },
    "motivation_message": "Struggling with loops is completely normal — every Python developer has been there. One more focused session will make it click."
}


# ── Agent Core ─────────────────────────────────────────────────────────────────
class StudyAgent:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client  = anthropic.Anthropic(api_key=self.api_key) if HAS_ANTHROPIC and self.api_key else None
        self.history = []   # simple in-session memory

    def analyze(self, study_log: str) -> dict:
        """Send study log to LLM and parse the JSON response."""
        self.history.append({"role": "user", "content": study_log})

        if self.client:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=self.history,
            )
            raw = response.content[0].text
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"error": "Could not parse LLM response", "raw": raw}
        else:
            # Mock mode
            result = MOCK_RESPONSE

        self.history.append({"role": "assistant", "content": json.dumps(result)})
        return result

    def format_output(self, result: dict) -> str:
        """Pretty-print the agent response for the terminal."""
        if "error" in result:
            return f"❌ Error: {result['error']}\n\nRaw:\n{result.get('raw','')}"

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("🤖 STUDY AGENT REPORT")
        lines.append("=" * 60)

        lines.append(f"\n📝 Summary:\n   {result.get('summary','')}")

        lines.append("\n✅ Strengths:")
        for s in result.get("strengths", []):
            lines.append(f"   • {s}")

        lines.append("\n⚠️  Areas to Improve:")
        for w in result.get("weaknesses", []):
            lines.append(f"   • {w}")

        plan = result.get("next_session_plan", {})
        lines.append(f"\n📅 Next Session Plan ({plan.get('estimated_time_minutes','?')} min):")
        lines.append(f"   Focus: {plan.get('focus_topic','')}")
        lines.append("   Activities:")
        for act in plan.get("recommended_activities", []):
            lines.append(f"     → {act}")

        lines.append(f"\n💬 {result.get('motivation_message','')}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PROJECT 4 — LLM-BASED STUDY AGENT")
    print("=" * 60)

    agent = StudyAgent()
    mode  = "LIVE (Anthropic API)" if agent.client else "MOCK (no API key)"
    print(f"   Mode: {mode}")

    # Example study log — student would type or paste this
    EXAMPLE_LOG = """
    Date: Today
    Study time: 2 hours
    Topics covered:
      - Python variables and data types (done, felt comfortable)
      - Functions and return values (done, understood well)
      - For loops (struggled — confused about range() and nested loops)
      - List comprehensions (attempted but syntax feels strange)

    Quizzes: Scored 6/10 on loops quiz, 9/10 on functions quiz
    Homework: Completed 3 out of 5 exercises
    Notes: I keep making off-by-one errors in loops. Comprehensions look cool but hard to read.
    Goal: Finish Python basics this week before moving to NumPy.
    """

    print("\n📖 Analyzing study log...")
    result = agent.analyze(EXAMPLE_LOG)
    print(agent.format_output(result))

    # Optional: interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit)")
    print("   Paste your own study log, then press Enter twice.\n")
    while True:
        lines = []
        print("Your log (press Enter twice to submit):")
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        user_log = "\n".join(lines).strip()
        if not user_log or user_log.lower() in ("quit", "exit"):
            print("Goodbye! Keep studying! 🎓")
            break
        result = agent.analyze(user_log)
        print(agent.format_output(result))


if __name__ == "__main__":
    main()
