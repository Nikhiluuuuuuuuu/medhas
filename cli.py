#!/usr/bin/env python3
"""
Medhas AGI Memory Engine - Standalone Interactive CLI
Usage:
  Interactive Mode: python cli.py
  Single Command:   python cli.py remember "Alice Smith leads Project Titan."
                    python cli.py recall "Who leads Project Titan?"
"""

import sys
import os
import argparse

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from medhas.core import MedhasMemoryCore
from medhas.consolidation.ebbinghaus import EbbinghausScrubber

def print_banner():
    print("=" * 68)
    print("  🧠 MEDHAS AGI MEMORY ENGINE - INTERACTIVE CLI (v1.0.0-PROD)")
    print("  Universal Multimodal Bi-Temporal Memory Engine for Autonomous AI")
    print("=" * 68)

def interactive_shell(memory: MedhasMemoryCore):
    print_banner()
    print("\nAvailable Commands:")
    print("  remember <text>  - Ingest raw natural language text into long-term memory")
    print("  recall <query>   - Perform sub-100ms spreading activation context recall")
    print("  paste            - Paste multi-line raw paragraph text to remember")
    print("  decay            - Run offline Ebbinghaus memory decay scrubber")
    print("  stats            - View current graph node and edge counts")
    print("  exit / quit      - Exit the Medhas CLI\n")

    while True:
        try:
            user_input = input("medhas> ").strip()
            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                print("Exiting Medhas CLI. Goodbye!")
                break

            elif user_input.startswith("remember "):
                text = user_input[9:].strip()
                do_remember(memory, text)

            elif user_input.startswith("recall "):
                query = user_input[7:].strip()
                do_recall(memory, query)

            elif cmd_lower == "paste":
                print("Enter/Paste multi-line text (type 'END' on a new line when done):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                raw_text = "\n".join(lines).strip()
                if raw_text:
                    do_remember(memory, raw_text)

            elif cmd_lower == "decay":
                print("\n[MAINTENANCE] Running Ebbinghaus Memory Decay Scrubber...")
                scrubber = EbbinghausScrubber()
                scrubber.run_scrubber(memory.kuzu)
                print("[OK] Ebbinghaus maintenance complete. Low salience edges pruned.\n")

            elif cmd_lower == "stats":
                df_nodes = memory.kuzu.execute("MATCH (e:Entity) RETURN count(e) AS cnt").get_as_df()
                df_rels = memory.kuzu.execute("MATCH (a:Entity)-[r:CONNECTS]->(b:Entity) WHERE r.valid_to = 0.0 RETURN count(r) AS cnt").get_as_df()
                node_cnt = df_nodes['cnt'].iloc[0] if not df_nodes.empty else 0
                rel_cnt = df_rels['cnt'].iloc[0] if not df_rels.empty else 0
                print(f"\n[GRAPH STATS] Total Active Entities: {node_cnt} | Active Bi-Temporal Edges: {rel_cnt}\n")

            else:
                # Treat un-prefixed input as remember text
                do_remember(memory, user_input)

        except KeyboardInterrupt:
            print("\nExiting Medhas CLI.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

def do_remember(memory: MedhasMemoryCore, text: str):
    print(f"\n[INGESTING RAW TEXT] \"{text}\"")
    ingested = memory.remember_raw_text(text)
    if not ingested:
        print("[WARN] Processed text, but no distinct entity relationships were identified.")
    else:
        print(f"[OK] Successfully stored {len(ingested)} memory graph links:")
        for src, rel, tgt in ingested:
            print(f"   • ({src}) --[{rel}]--> ({tgt})")
    print()

def do_recall(memory: MedhasMemoryCore, query: str):
    print(f"\n[RECALLING CONTEXT FOR QUERY] \"{query}\"")
    result = memory.recall(query)
    print("\n" + result + "\n")

def main():
    parser = argparse.ArgumentParser(description="Medhas AGI Memory Core CLI")
    parser.add_argument("command", nargs="?", choices=["remember", "recall", "decay", "stats"], help="Command to run")
    parser.add_argument("text", nargs="*", help="Raw text or query string")
    parser.add_argument("--db", default="./medhas_db", help="Path to KuzuDB database directory")
    parser.add_argument("--wal", default="./medhas_wal.db", help="Path to SQLite WAL database file")

    args = parser.parse_args()
    memory = MedhasMemoryCore(db_path=args.db, wal_path=args.wal)

    if not args.command:
        interactive_shell(memory)
    else:
        arg_text = " ".join(args.text).strip()
        if args.command == "remember":
            if not arg_text:
                print("Error: Please provide text to remember.")
                sys.exit(1)
            do_remember(memory, arg_text)
        elif args.command == "recall":
            if not arg_text:
                print("Error: Please provide query to recall.")
                sys.exit(1)
            do_recall(memory, arg_text)
        elif args.command == "decay":
            scrubber = EbbinghausScrubber()
            scrubber.run_scrubber(memory.kuzu)
            print("✓ Ebbinghaus maintenance complete.")
        elif args.command == "stats":
            df_nodes = memory.kuzu.execute("MATCH (e:Entity) RETURN count(e) AS cnt").get_as_df()
            df_rels = memory.kuzu.execute("MATCH (a:Entity)-[r:CONNECTS]->(b:Entity) WHERE r.valid_to = 0.0 RETURN count(r) AS cnt").get_as_df()
            print(f"Entities: {df_nodes['cnt'].iloc[0]} | Active Edges: {df_rels['cnt'].iloc[0]}")

if __name__ == "__main__":
    main()
