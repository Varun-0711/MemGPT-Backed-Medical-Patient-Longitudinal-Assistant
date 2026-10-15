"""One process == one session. Run `python -m medassist.cli --db app.db --user patient_001`."""
import argparse

from .agent import Agent
from .seed import seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="medassist.db")
    p.add_argument("--user", default="patient_001")
    p.add_argument("--seed", action="store_true", help="create demo profile + archival history if missing")
    p.add_argument("--say", action="append", help="non-interactive patient message (repeatable)")
    p.add_argument("--show-core", action="store_true")
    a = p.parse_args()

    if a.seed:
        seed(a.db, a.user)
    agent = Agent(a.db, a.user)
    print(f"[boot] session={agent.session_id} llm={agent.llm.name} "
          f"hydrated_blocks={list(agent.core.blocks)} context_messages=0 "
          f"ctx_tokens={agent.context_tokens()}/{agent.TOKEN_BUDGET}")
    if a.show_core:
        print("[core human_block]\n" + agent.core.blocks["human_block"])

    def turn(text):
        r = agent.step(text)
        print(f"PATIENT> {text}")
        for t in r["tools"]:
            print(f"   tool: {t}")
        print(f"ASSISTANT> {r['reply']}")
        print(f"[ctx_tokens={r['tokens']}/{agent.TOKEN_BUDGET} paged_msgs={agent.paged_total}]")

    if a.say:
        for text in a.say:
            turn(text)
    else:
        print("Type a message (Ctrl-D to quit).")
        try:
            while True:
                turn(input("you> "))
        except EOFError:
            pass


if __name__ == "__main__":
    main()
