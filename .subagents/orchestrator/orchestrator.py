import argparse
import json
import subprocess


class Agent:
    def __init__(self, name: str, description: str, commands: dict[str, str]):
        self.name = name
        self.description = description
        self.commands = commands
        self.session_id = None

    def start_task(self, task: str) -> str:
        """Starts a task with the agent and returns the initial response."""
        command_template = self.commands.get("start")
        if not command_template:
            raise ValueError(f"No start command for agent {self.name}")

        # Replace placeholders
        # Note: robust implementation would use a proper template engine or safer substitution
        command = command_template.replace("$TASK", task).replace("$SESSION_ID", "")

        print(f"[{self.name}] Starting task: {task[:50]}...")
        return self._execute_command(command)

    def resume_task(self, answer: str) -> str:
        """Resumes the task with the agent."""
        if not self.session_id:
            raise ValueError(f"Agent {self.name} has no active session to resume.")

        command_template = self.commands.get("resume")
        if not command_template:
            raise ValueError(f"No resume command for agent {self.name}")

        command = command_template.replace("$SESSION_ID", self.session_id).replace(
            "$ANSWER", answer
        )

        print(f"[{self.name}] Resuming session {self.session_id}...")
        return self._execute_command(command)

    def _execute_command(self, command_str: str) -> str:
        """Executes the shell command and captures output."""
        # On Windows effectively, we might need shell=True or specific executable handling
        # For this implementation, we assume the command string is formatted for the underlying shell
        try:
            # Using shell=True to handle variable expansion and system-specific execution
            # In a real production env, we'd want to avoid shell=True for security,
            # but the manifest commands are shell scripts.
            result = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            if result.returncode != 0:
                print(f"Error executing command: {result.stderr}")
                return f"Error: {result.stderr}"

            # Parse session ID if present in output (simplified parsing)
            # The scripts echo [SESSION_ID: xyz]
            for line in result.stdout.splitlines():
                if "[SESSION_ID:" in line:
                    self.session_id = line.split("[SESSION_ID:")[1].strip().strip("]")

            return result.stdout.strip()

        except Exception as e:
            return f"Exception executing command: {e!s}"


class Orchestrator:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.agents: dict[str, Agent] = {}
        self._load_manifest()

    def _load_manifest(self):
        with open(self.manifest_path, encoding="utf-8") as f:
            data = json.load(f)
            for agent_data in data.get("agents", []):
                self.agents[agent_data["name"]] = Agent(
                    name=agent_data["name"],
                    description=agent_data["description"],
                    commands=agent_data["commands"],
                )

    def list_agents(self):
        return list(self.agents.keys())

    def get_agent(self, name: str) -> Agent | None:
        return self.agents.get(name)

    def dispatch(self, agent_name: str, task: str):
        agent = self.get_agent(agent_name)
        if not agent:
            print(f"Agent {agent_name} not found.")
            return

        response = agent.start_task(task)
        print(f"\n--- Response from {agent_name} ---\n")
        print(response)
        print("\n-----------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Orchestrator")
    parser.add_argument(
        "--manifest", default=r"E:\chimera\.subagents\manifest.json", help="Path to manifest.json"
    )
    parser.add_argument("--agent", help="Specific agent to run")
    parser.add_argument("--task", help="Task for the agent")
    parser.add_argument("--list", action="store_true", help="List available agents")

    args = parser.parse_args()

    orchestrator = Orchestrator(args.manifest)

    if args.list:
        print("Available Agents:")
        for name in orchestrator.list_agents():
            print(f"- {name}")
        return

    if args.agent and args.task:
        orchestrator.dispatch(args.agent, args.task)
    elif args.task:
        # Simple routing logic (in future, use LLM to decide agent)
        print("Auto-routing not yet implemented. Please specify --agent.")
        print("Available agents:", orchestrator.list_agents())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
