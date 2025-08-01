#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generation Script
Generates comprehensive SBOM for supply chain security compliance
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def get_package_info() -> Dict:
    """Extract package information from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    return data.get("project", {})


def get_installed_packages() -> List[Dict]:
    """Get list of installed packages with versions."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "list", "--format=json"
        ], capture_output=True, text=True, check=True)
        
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting package list: {e}")
        return []


def get_dependency_tree() -> Dict:
    """Get dependency tree information."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "show", "--verbose", "reflexion-agent-boilerplate"
        ], capture_output=True, text=True)
        
        # Parse pip show output
        dependencies = []
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Requires:'):
                    deps = line.replace('Requires:', '').strip()
                    if deps:
                        dependencies = [dep.strip() for dep in deps.split(',')]
        
        return {"direct_dependencies": dependencies}
    except Exception as e:
        print(f"Error getting dependency tree: {e}")
        return {"direct_dependencies": []}


def get_git_info() -> Dict:
    """Get Git repository information."""
    try:
        # Get current commit hash
        commit_result = subprocess.run([
            "git", "rev-parse", "HEAD"
        ], capture_output=True, text=True, check=True)
        
        # Get current branch
        branch_result = subprocess.run([
            "git", "rev-parse", "--abbrev-ref", "HEAD"
        ], capture_output=True, text=True, check=True)
        
        # Get remote URL
        remote_result = subprocess.run([
            "git", "config", "--get", "remote.origin.url"
        ], capture_output=True, text=True)
        
        return {
            "commit": commit_result.stdout.strip(),
            "branch": branch_result.stdout.strip(),
            "remote_url": remote_result.stdout.strip() if remote_result.returncode == 0 else "unknown"
        }
    except subprocess.CalledProcessError:
        return {
            "commit": "unknown",
            "branch": "unknown", 
            "remote_url": "unknown"
        }


def generate_cyclone_dx_sbom() -> Dict:
    """Generate SBOM in CycloneDX format."""
    package_info = get_package_info()
    installed_packages = get_installed_packages()
    dependency_info = get_dependency_tree()
    git_info = get_git_info()
    
    # Create components list
    components = []
    for pkg in installed_packages:
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "scope": "required"
        }
        components.append(component)
    
    # Create main component (this package)
    main_component = {
        "type": "application",
        "bom-ref": f"pkg:pypi/{package_info.get('name', 'reflexion-agent-boilerplate')}@{package_info.get('version', '0.1.0')}",
        "name": package_info.get("name", "reflexion-agent-boilerplate"),
        "version": package_info.get("version", "0.1.0"),
        "description": package_info.get("description", ""),
        "licenses": [{"license": {"name": "Apache-2.0"}}] if package_info.get("license") else [],
        "purl": f"pkg:pypi/{package_info.get('name', 'reflexion-agent-boilerplate')}@{package_info.get('version', '0.1.0')}",
        "externalReferences": [
            {
                "type": "vcs",
                "url": git_info["remote_url"]
            }
        ] if git_info["remote_url"] != "unknown" else []
    }
    
    # Create SBOM structure
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:reflexion-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat() + "Z",
            "tools": [
                {
                    "vendor": "terragon-labs",
                    "name": "sbom-generator",
                    "version": "1.0.0"
                }
            ],
            "component": main_component
        },
        "components": components
    }
    
    # Add dependency relationships if available
    if dependency_info["direct_dependencies"]:
        dependencies = []
        main_ref = main_component["bom-ref"]
        
        for dep_name in dependency_info["direct_dependencies"]:
            # Find the dependency in installed packages
            for pkg in installed_packages:
                if pkg["name"].lower() == dep_name.lower():
                    dependencies.append({
                        "ref": main_ref,
                        "dependsOn": [f"pkg:pypi/{pkg['name']}@{pkg['version']}"]
                    })
                    break
        
        if dependencies:
            sbom["dependencies"] = dependencies
    
    return sbom


def generate_spdx_sbom() -> Dict:
    """Generate SBOM in SPDX format."""
    package_info = get_package_info()
    installed_packages = get_installed_packages()
    git_info = get_git_info()
    
    # Create SPDX document
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{package_info.get('name', 'reflexion-agent-boilerplate')}-sbom",
        "documentNamespace": f"https://github.com/your-org/reflexion-agent-boilerplate/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "creationInfo": {
            "created": datetime.now().isoformat() + "Z",
            "creators": ["Tool: terragon-sbom-generator-1.0.0"]
        },
        "packages": []
    }
    
    # Add main package
    main_package = {
        "SPDXID": "SPDXRef-Package",
        "name": package_info.get("name", "reflexion-agent-boilerplate"),
        "downloadLocation": git_info.get("remote_url", "NOASSERTION"),
        "filesAnalyzed": False,
        "versionInfo": package_info.get("version", "0.1.0"),
        "supplier": "Organization: Terragon Labs",
        "copyrightText": "Apache-2.0"
    }
    sbom["packages"].append(main_package)
    
    # Add dependencies
    for i, pkg in enumerate(installed_packages):
        package = {
            "SPDXID": f"SPDXRef-Package-{i+1}",
            "name": pkg["name"],
            "downloadLocation": f"https://pypi.org/project/{pkg['name']}/",
            "filesAnalyzed": False,
            "versionInfo": pkg["version"],
            "supplier": "NOASSERTION",
            "copyrightText": "NOASSERTION"
        }
        sbom["packages"].append(package)
    
    return sbom


def main():
    """Generate SBOM files."""
    output_dir = Path(".")
    
    # Generate CycloneDX SBOM
    try:
        cyclone_sbom = generate_cyclone_x_sbom()
        cyclone_path = output_dir / "sbom-cyclonedx.json"
        with open(cyclone_path, 'w') as f:
            json.dump(cyclone_sbom, f, indent=2)
        print(f"Generated CycloneDX SBOM: {cyclone_path}")
    except Exception as e:
        print(f"Error generating CycloneDX SBOM: {e}")
    
    # Generate SPDX SBOM
    try:
        spdx_sbom = generate_spdx_sbom()
        spdx_path = output_dir / "sbom-spdx.json"
        with open(spdx_path, 'w') as f:
            json.dump(spdx_sbom, f, indent=2)
        print(f"Generated SPDX SBOM: {spdx_path}")
    except Exception as e:
        print(f"Error generating SPDX SBOM: {e}")
    
    # Also generate a simple combined SBOM
    try:
        combined_sbom = {
            "format": "terragon-sbom-v1",
            "generated_at": datetime.now().isoformat(),
            "package": get_package_info(),
            "git_info": get_git_info(),
            "dependencies": get_installed_packages(),
            "dependency_tree": get_dependency_tree()
        }
        
        combined_path = output_dir / "sbom.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_sbom, f, indent=2)
        print(f"Generated combined SBOM: {combined_path}")
    except Exception as e:
        print(f"Error generating combined SBOM: {e}")


if __name__ == "__main__":
    main()