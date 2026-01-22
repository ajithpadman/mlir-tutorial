"""CLI interface for mlirdialectgen."""

import os
import sys
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Get the directory where this package is installed
PACKAGE_DIR = Path(__file__).parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"


def get_llvm_install_dir() -> str:
    """Get LLVM_INSTALL_DIR from environment or default."""
    llvm_dir = os.environ.get("LLVM_INSTALL_DIR")
    if not llvm_dir:
        # Default to ../llvm-install relative to current directory
        llvm_dir = str(Path.cwd() / ".." / "llvm-install")
    return llvm_dir


def normalize_dialect_name(name: str) -> tuple[str, str]:
    """Convert dialect name to various formats.
    
    Returns:
        (dialect_name, DialectName): lowercase name and PascalCase name
    """
    # Convert to lowercase for dialect name
    dialect_name = name.lower().replace("-", "").replace("_", "")
    
    # Convert to PascalCase for C++ class names
    parts = name.replace("-", "_").split("_")
    DialectName = "".join(part.capitalize() for part in parts)
    
    return dialect_name, DialectName


@click.command()
@click.argument("dialect_name", required=True)
@click.option(
    "--llvm-install-dir",
    default=None,
    help="Path to LLVM installation directory (defaults to $LLVM_INSTALL_DIR or ../llvm-install)",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
)
@click.option(
    "--output-dir",
    default=None,
    help="Output directory for generated dialect (defaults to current directory)",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
)
def main(dialect_name: str, llvm_install_dir: str | None, output_dir: str | None):
    """Generate boilerplate for a standalone MLIR dialect.
    
    DIALECT_NAME: Name of the dialect (e.g., 'my-dialect' or 'mydialect')
    """
    # Normalize dialect name
    dialect_lower, DialectName = normalize_dialect_name(dialect_name)
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / dialect_lower
    else:
        output_dir = Path(output_dir)
    
    # Get LLVM install directory
    if llvm_install_dir is None:
        llvm_install_dir = get_llvm_install_dir()
    
    # Setup Jinja2 environment
    if not TEMPLATES_DIR.exists():
        click.echo(f"Error: Templates directory not found: {TEMPLATES_DIR}", err=True)
        sys.exit(1)
    
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    # Template context
    context = {
        "dialect_name": dialect_lower,
        "DialectName": DialectName,
        "DIALECT_NAME": dialect_lower.upper().replace("-", "_"),
        "llvm_install_dir": llvm_install_dir,
    }
    
    # List of files to generate
    files_to_generate = [
        ("CMakeLists.txt", "root"),
        ("include/CMakeLists.txt", "include"),
        (f"include/{DialectName}/CMakeLists.txt", "include_dialect"),
        (f"include/{DialectName}/{DialectName}Dialect.td", "dialect_td"),
        (f"include/{DialectName}/{DialectName}Ops.td", "ops_td"),
        (f"include/{DialectName}/{DialectName}Ops.h", "ops_h"),
        (f"include/{DialectName}/{DialectName}Dialect.h", "dialect_h"),
        ("lib/CMakeLists.txt", "lib"),
        (f"lib/{DialectName}/CMakeLists.txt", "lib_dialect"),
        (f"lib/{DialectName}/{DialectName}Dialect.cpp", "dialect_cpp"),
        (f"lib/{DialectName}/{DialectName}Ops.cpp", "ops_cpp"),
        ("README.md", "readme"),
    ]
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Generating MLIR dialect '{dialect_lower}' in {output_dir}")
    
    # Generate files
    for file_path, template_name in files_to_generate:
        full_path = output_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            template = env.get_template(f"{template_name}.jinja")
            content = template.render(**context)
            
            with open(full_path, "w") as f:
                f.write(content)
            
            click.echo(f"  Created: {file_path}")
        except Exception as e:
            click.echo(f"  Error generating {file_path}: {e}", err=True)
            sys.exit(1)
    
    click.echo(f"\n✓ Dialect '{dialect_lower}' generated successfully!")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. cd {output_dir}")
    click.echo(f"  2. mkdir build && cd build")
    click.echo(f"  3. cmake .. -DLLVM_INSTALL_DIR={llvm_install_dir}")
    click.echo(f"  4. make -j$(nproc)")


if __name__ == "__main__":
    main()

