import sys
import unittest
from unittest.mock import patch

# Add the directory containing orchestrator.py to the path
sys.path.append(r"E:\chimera\.subagents\orchestrator")

from orchestrator import Orchestrator


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.manifest_path = r"E:\chimera\.subagents\manifest.json"
        self.orchestrator = Orchestrator(self.manifest_path)

    def test_load_manifest(self):
        """Verify manifest loads and contains expected agents."""
        agents = self.orchestrator.list_agents()
        expected = ["backend-architect", "frontend-architect", "qa-engineer", "devops-engineer"]
        for agent in expected:
            self.assertIn(agent, agents)

    @patch("subprocess.run")
    def test_agent_start_command(self, mock_run):
        """Verify start command formatting."""
        mock_run.return_value.stdout = "[SESSION_ID: 12345]\nTask started."
        mock_run.return_value.returncode = 0

        agent = self.orchestrator.get_agent("frontend-architect")
        agent.start_task("Build a button")

        # Verify command contains key elements
        args, _ = mock_run.call_args
        command_str = args[0]
        self.assertIn("powershell", command_str)
        self.assertIn("start.ps1", command_str)
        self.assertIn("frontend-architect", command_str)
        self.assertIn("Build a button", command_str)

    @patch("subprocess.run")
    def test_agent_resume_command(self, mock_run):
        """Verify resume command formatting."""
        mock_run.return_value.stdout = "Task resumed."
        mock_run.return_value.returncode = 0

        agent = self.orchestrator.get_agent("qa-engineer")
        agent.session_id = "test-session-id"
        agent.resume_task("Here is the test plan")

        args, _ = mock_run.call_args
        command_str = args[0]
        self.assertIn("powershell", command_str)
        self.assertIn("resume.ps1", command_str)
        self.assertIn("test-session-id", command_str)
        self.assertIn("qa-engineer", command_str)


if __name__ == "__main__":
    unittest.main()
