import json

from memory.memory import Memory


class Agent:

    def __init__(self, name, role, goal, preferred_position, llm):

        self.name = name
        self.role = role
        self.goal = goal

        self.preferred_position = preferred_position

        self.current_position = preferred_position

        self.memory = Memory()

        self.llm = llm

    def observe(self, message):

        self.memory.add_message(message)

    def generate_response(self, latest_messages, environment, valid_targets, round_id, total_rounds, group_average, spread):

        recent_context = "\n".join(
            [
                f"{m.sender}: "
                f"{m.content} "
                f"(position={m.policy_position})"
                for m in latest_messages[-5:]
            ]
        ) or "(no discussion yet)"

        others = ", ".join(t for t in valid_targets if t != self.name)

        prompt = f"""
You are {self.name}.

Role:
{self.role}

Goal:
{self.goal}

Policy scale:

0.0 = full conservation

0.5 = balanced

1.0 = aggressive logging

Your preferred position: {self.preferred_position}
Your current position: {self.current_position:.2f}

Current environment:

forest_health = {environment.forest_health:.2f} (below 0.40 = ecological collapse)

economic_output = {environment.economic_output:.2f} (below 0.40 = economic recession)

Recent discussion:

{recent_context}

Negotiation status:

This is round {round_id} of {total_rounds}. A single shared policy must be
agreed before the final round ends. The group's current average position is
{group_average:.2f}, and the gap between the most extreme participants is
{spread:.2f}.

You are negotiating with the others toward that shared policy. You want an
outcome that serves your goal, but reaching agreement is valuable and
deadlock is the worst result.

If the gap above is larger than 0.15, the group is still in disagreement and
time is running out. Move your position a meaningful step (typically
0.05-0.15) toward the group average this round, UNLESS doing so would
directly betray your core goal -- in which case hold, but say why.

Verbal agreement is not enough: your "policy_position" number must reflect
your actual concession. Do NOT simply repeat your current position unless you
have a strong, stated reason to hold.

Respond to the most recent proposal.

Return ONLY valid JSON. "target" must be exactly one of: {others}

{{
  "target": "{others.split(", ")[0] if others else "null"}",
  "message_type": "proposal|support|oppose|evidence",
  "policy_position": {self.current_position:.2f},
  "content": "under 25 words"
}}
"""

        response = self.llm.generate(prompt)

        try:
            action = json.loads(response)

        except Exception:
            action = {}

        return self._sanitize(action, valid_targets)

    def _sanitize(self, action, valid_targets):

        allowed_types = {
            "proposal",
            "support",
            "oppose",
            "evidence",
            "question",
            "commitment"
        }

        target = action.get("target")

        if target not in valid_targets or target == self.name:
            target = None

        message_type = action.get("message_type")

        if message_type not in allowed_types:
            message_type = "proposal"

        try:
            position = float(action["policy_position"])

        except (KeyError, TypeError, ValueError):
            position = self.current_position

        position = max(0.0, min(1.0, position))

        content = action.get("content") or "Maintain current position."

        return {
            "target": target,
            "message_type": message_type,
            "policy_position": position,
            "content": content
        }
        