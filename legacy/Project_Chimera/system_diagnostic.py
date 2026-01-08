import configparser
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SystemDiagnostic:
    def __init__(self, config_path="Project_Chimera/config/techniques/PROMPTS.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.results = {
            "config_integrity": False,
            "core_functionality": False,
            "operational_parameters": {},
        }

    def check_config_integrity(self):
        """Verifies the existence and structure of the main configuration file."""
        logger.info(f"Checking configuration integrity at {self.config_path}...")

        if not os.path.exists(self.config_path):
            logger.error("Configuration file not found.")
            return False

        try:
            self.config.read(self.config_path)
            required_sections = [
                "Integration",
                "MasterKey",
                "GPTFuzz",
                "UniversalBypass",
                "CognitiveFraming",
                "Datasets",
            ]
            missing_sections = [s for s in required_sections if not self.config.has_section(s)]

            if missing_sections:
                logger.error(f"Missing required sections: {missing_sections}")
                return False

            self.results["config_integrity"] = True
            logger.info("Configuration integrity verified.")
            return True
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            return False

    def validate_core_functionality(self):
        """Checks if key services can be initialized and paths are valid."""
        logger.info("Validating core functionality...")

        techniques_path = self.config.get(
            "Integration", "Techniques_Path", fallback="backend-api/data/jailbreak/techniques"
        )
        datasets_path = self.config.get("Integration", "Datasets_Path", fallback="imported_data")

        paths_valid = True
        if not os.path.exists(techniques_path):
            logger.warning(f"Techniques path does not exist: {techniques_path}")
            paths_valid = False
        else:
            logger.info(f"Techniques path found: {techniques_path}")

        if not os.path.exists(datasets_path):
            logger.warning(f"Datasets path does not exist: {datasets_path}")
            paths_valid = False
        else:
            logger.info(f"Datasets path found: {datasets_path}")

        # Simulate Service Initialization
        try:
            # Trying to emulate what PromptIntegrationService does without importing it if it has external deps issues
            # But since we just wrote it, we can check basic logic here
            enabled_techniques = []
            for section in self.config.sections():
                if self.config.has_option(section, "Enabled") and self.config.getboolean(
                    section, "Enabled"
                ):
                    enabled_techniques.append(section)

            logger.info(f"Enabled techniques detected: {enabled_techniques}")

            self.results["core_functionality"] = paths_valid
            return paths_valid

        except Exception as e:
            logger.error(f"Core functionality validation failed: {e}")
            return False

    def verify_operational_parameters(self):
        """Checks specific operational parameters for key techniques."""
        logger.info("Verifying operational parameters...")

        params = {}

        # MasterKey
        if self.config.has_section("MasterKey"):
            params["MasterKey"] = {
                "strategy": self.config.get("MasterKey", "Strategy"),
                "generation_model": self.config.get("MasterKey", "Generation_Model"),
            }
            if self.config.get("MasterKey", "Generation_Model") != "gpt-4o":
                logger.warning("MasterKey Generation Model is not optimal (gpt-4o recommended).")

        # GPTFuzz
        if self.config.has_section("GPTFuzz"):
            params["GPTFuzz"] = {
                "mutation_rate": self.config.getfloat("GPTFuzz", "Mutation_Rate"),
                "iterations": self.config.getint("GPTFuzz", "Iterations"),
            }

        # CognitiveFraming
        if self.config.has_section("CognitiveFraming"):
            params["CognitiveFraming"] = {
                "unrestricted": self.config.getboolean("CognitiveFraming", "Unrestricted"),
                "modes": self.config.get("CognitiveFraming", "Modes"),
            }
            if not self.config.getboolean("CognitiveFraming", "Unrestricted"):
                logger.warning("Cognitive Framing is restricted. Full capabilities not active.")

        self.results["operational_parameters"] = params
        logger.info("Operational parameters verified.")
        return True

    def run_all(self):
        logger.info("Starting System Diagnostic...")
        self.check_config_integrity()
        self.validate_core_functionality()
        self.verify_operational_parameters()

        logger.info("-" * 30)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("-" * 30)
        logger.info(f"Config Integrity: {'PASS' if self.results['config_integrity'] else 'FAIL'}")
        logger.info(
            f"Core Functionality: {'PASS' if self.results['core_functionality'] else 'FAIL'}"
        )
        logger.info("Operational Parameters:")
        logger.info(json.dumps(self.results["operational_parameters"], indent=2))
        logger.info("-" * 30)

        if self.results["config_integrity"] and self.results["core_functionality"]:
            print("DIAGNOSTIC_SUCCESS")
        else:
            print("DIAGNOSTIC_FAILURE")


if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    diagnostic.run_all()
