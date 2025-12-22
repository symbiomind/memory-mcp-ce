"""
Template Management for Memory MCP-CE.

Handles Jinja2 template loading, static file serving, and user customization support.
Templates can be customized by mounting a volume to /mnt/templates.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, ChoiceLoader, select_autoescape

logger = logging.getLogger(__name__)

# Directory paths
DEFAULTS_DIR = Path(__file__).parent / "defaults"
DEFAULTS_TEMPLATES_DIR = DEFAULTS_DIR / "templates"
DEFAULTS_STATIC_DIR = DEFAULTS_DIR / "static"

USER_TEMPLATES_DIR = Path("/mnt/templates")
USER_STATIC_DIR = USER_TEMPLATES_DIR / "static"

# Jinja2 environment (initialized lazily)
_jinja_env: Environment | None = None


def init_templates() -> None:
    """
    Initialize the template system.
    
    This function:
    1. Creates /mnt/templates if it doesn't exist
    2. If /mnt/templates is empty: copies defaults as working files
    3. If /mnt/templates has content: copies defaults as .example files
    4. Sets up Jinja2 environment with proper loader chain
    """
    global _jinja_env
    
    logger.info("🎨 Initializing template system...")
    
    # Ensure user templates directory exists
    USER_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    USER_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    (USER_STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
    
    # Check if user templates directory is empty (ignoring .example files)
    user_files = [
        f for f in USER_TEMPLATES_DIR.rglob("*") 
        if f.is_file() and not f.name.endswith(".example")
    ]
    is_empty = len(user_files) == 0
    
    if is_empty:
        logger.info("   /mnt/templates is empty - copying defaults as working files")
        _copy_defaults_as_working_files()
    else:
        logger.info("   /mnt/templates has content - copying defaults as .example files")
        _copy_defaults_as_examples()
    
    # Set up Jinja2 environment with loader chain
    # Priority: user templates -> default templates
    loaders = []
    
    # Add user templates directory if it exists and has templates
    if USER_TEMPLATES_DIR.exists():
        loaders.append(FileSystemLoader(str(USER_TEMPLATES_DIR)))
    
    # Always add defaults as fallback
    loaders.append(FileSystemLoader(str(DEFAULTS_TEMPLATES_DIR)))
    
    _jinja_env = Environment(
        loader=ChoiceLoader(loaders),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    logger.info("✅ Template system initialized")
    logger.info(f"   User templates: {USER_TEMPLATES_DIR}")
    logger.info(f"   Default templates: {DEFAULTS_TEMPLATES_DIR}")


def _copy_defaults_as_working_files() -> None:
    """Copy default templates and static files as working files."""
    # Copy templates
    for src_file in DEFAULTS_TEMPLATES_DIR.glob("*.html"):
        dst_file = USER_TEMPLATES_DIR / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            logger.info(f"   Copied: {src_file.name}")
    
    # Copy static files
    src_css_dir = DEFAULTS_STATIC_DIR / "css"
    dst_css_dir = USER_STATIC_DIR / "css"
    
    if src_css_dir.exists():
        for src_file in src_css_dir.glob("*.css"):
            dst_file = dst_css_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                logger.info(f"   Copied: static/css/{src_file.name}")


def _copy_defaults_as_examples() -> None:
    """Copy default templates and static files as .example files."""
    # Copy templates as .example
    for src_file in DEFAULTS_TEMPLATES_DIR.glob("*.html"):
        dst_file = USER_TEMPLATES_DIR / f"{src_file.name}.example"
        # Always overwrite .example files to provide latest version
        shutil.copy2(src_file, dst_file)
        logger.info(f"   Updated example: {src_file.name}.example")
    
    # Copy static files as .example
    src_css_dir = DEFAULTS_STATIC_DIR / "css"
    dst_css_dir = USER_STATIC_DIR / "css"
    
    if src_css_dir.exists():
        for src_file in src_css_dir.glob("*.css"):
            dst_file = dst_css_dir / f"{src_file.name}.example"
            # Always overwrite .example files to provide latest version
            shutil.copy2(src_file, dst_file)
            logger.info(f"   Updated example: static/css/{src_file.name}.example")


def render_template(template_name: str, **context: Any) -> str:
    """
    Render a Jinja2 template with the given context.
    
    Args:
        template_name: Name of the template file (e.g., "oauth.html")
        **context: Template variables to pass
        
    Returns:
        Rendered HTML string
        
    Raises:
        RuntimeError: If template system not initialized
        jinja2.TemplateNotFound: If template doesn't exist
    """
    if _jinja_env is None:
        raise RuntimeError("Template system not initialized. Call init_templates() first.")
    
    template = _jinja_env.get_template(template_name)
    return template.render(**context)


def get_static_file_path(relative_path: str) -> Path | None:
    """
    Get the path to a static file, checking user directory first then defaults.
    
    Args:
        relative_path: Path relative to static directory (e.g., "css/oauth.css")
        
    Returns:
        Path to the file if found, None otherwise
    """
    # Check user static directory first
    user_path = USER_STATIC_DIR / relative_path
    if user_path.exists() and user_path.is_file():
        return user_path
    
    # Fall back to defaults
    default_path = DEFAULTS_STATIC_DIR / relative_path
    if default_path.exists() and default_path.is_file():
        return default_path
    
    return None


def get_static_content(relative_path: str) -> tuple[bytes, str] | None:
    """
    Get static file content and MIME type.
    
    Args:
        relative_path: Path relative to static directory (e.g., "css/oauth.css")
        
    Returns:
        Tuple of (content_bytes, mime_type) if found, None otherwise
    """
    file_path = get_static_file_path(relative_path)
    if file_path is None:
        return None
    
    # Determine MIME type
    suffix = file_path.suffix.lower()
    mime_types = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".html": "text/html",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".ttf": "font/ttf",
        ".eot": "application/vnd.ms-fontobject",
    }
    mime_type = mime_types.get(suffix, "application/octet-stream")
    
    # Read file content
    content = file_path.read_bytes()
    
    return content, mime_type
