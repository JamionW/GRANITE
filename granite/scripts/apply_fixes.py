"""
fix_indentation.py

Safe script to fix indentation issues without modifying logic.
"""

import os
import ast
import autopep8  # You may need to: pip install autopep8


def check_file_syntax(filepath):
    """Check if a Python file has syntax errors and report them."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        ast.parse(content)
        print(f"✓ {filepath}: No syntax errors")
        return True, None
    except SyntaxError as e:
        print(f"✗ {filepath}: Syntax error at line {e.lineno}")
        print(f"  Error: {e.msg}")
        if e.text:
            print(f"  Problem line: {e.text.strip()}")
        return False, e
    except IndentationError as e:
        print(f"✗ {filepath}: Indentation error at line {e.lineno}")
        print(f"  Error: {e.msg}")
        return False, e


def fix_mixed_indentation(filepath):
    """Convert all tabs to spaces and ensure consistent indentation."""
    print(f"\nFixing indentation in {filepath}...")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Convert tabs to 4 spaces (Python standard)
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        # Replace tabs with 4 spaces
        fixed_line = line.replace('\t', '    ')
        fixed_lines.append(fixed_line)
        
        if '\t' in line:
            print(f"  Line {i}: Converted tabs to spaces")
    
    # Write back
    temp_file = filepath + '.fixing'
    with open(temp_file, 'w') as f:
        f.writelines(fixed_lines)
    
    # Check if the fix worked
    is_valid, error = check_file_syntax(temp_file)
    
    if is_valid:
        # Replace original with fixed version
        os.replace(temp_file, filepath)
        print(f"✓ Successfully fixed {filepath}")
        return True
    else:
        # Remove temp file
        os.remove(temp_file)
        print(f"✗ Fix didn't resolve all issues in {filepath}")
        return False


def auto_fix_with_autopep8(filepath):
    """Use autopep8 to automatically fix indentation issues."""
    try:
        print(f"\nTrying autopep8 on {filepath}...")
        
        with open(filepath, 'r') as f:
            original_content = f.read()
        
        # Fix with autopep8
        fixed_content = autopep8.fix_code(
            original_content,
            options={'aggressive': 1, 'max_line_length': 100}
        )
        
        # Write to temp file first
        temp_file = filepath + '.autopep8'
        with open(temp_file, 'w') as f:
            f.write(fixed_content)
        
        # Check if it's valid
        is_valid, _ = check_file_syntax(temp_file)
        
        if is_valid:
            os.replace(temp_file, filepath)
            print(f"✓ autopep8 successfully fixed {filepath}")
            return True
        else:
            os.remove(temp_file)
            print(f"✗ autopep8 couldn't fix all issues in {filepath}")
            return False
            
    except ImportError:
        print("autopep8 not installed. Run: pip install autopep8")
        return False
    except Exception as e:
        print(f"Error using autopep8: {e}")
        return False


def restore_from_backup_if_exists(filepath):
    """Restore from backup if available."""
    import glob
    
    backup_pattern = f"{filepath}.backup_*"
    backups = glob.glob(backup_pattern)
    
    if backups:
        # Get most recent backup
        latest_backup = max(backups, key=os.path.getmtime)
        
        # Check if backup is valid
        is_valid, _ = check_file_syntax(latest_backup)
        
        if is_valid:
            import shutil
            shutil.copy2(latest_backup, filepath)
            print(f"✓ Restored {filepath} from {latest_backup}")
            return True
        else:
            print(f"✗ Backup {latest_backup} also has syntax errors")
    
    return False


def main():
    """Main fixing routine."""
    print("="*60)
    print("Python Indentation Fixer for GRANITE")
    print("="*60)
    
    # Files to check
    files_to_fix = [
        "granite/baselines/idm.py",
        "granite/disaggregation/pipeline.py"
    ]
    
    print("\n1. Initial syntax check...")
    files_with_errors = []
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            is_valid, error = check_file_syntax(filepath)
            if not is_valid:
                files_with_errors.append(filepath)
    
    if not files_with_errors:
        print("\n✓ All files have valid syntax!")
        return
    
    print(f"\n2. Found {len(files_with_errors)} files with errors")
    
    for filepath in files_with_errors:
        print(f"\n--- Fixing {filepath} ---")
        
        # Try restoring from backup first
        if restore_from_backup_if_exists(filepath):
            continue
        
        # Try simple tab-to-space conversion
        if fix_mixed_indentation(filepath):
            continue
        
        # Try autopep8
        if auto_fix_with_autopep8(filepath):
            continue
        
        print(f"\n✗ Could not automatically fix {filepath}")
        print("  Manual intervention needed. The issue might be:")
        print("  1. Missing colons after def/if/for/while statements")
        print("  2. Mismatched parentheses or brackets")
        print("  3. Incorrect indentation levels")
    
    print("\n3. Final syntax check...")
    all_valid = True
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            is_valid, _ = check_file_syntax(filepath)
            if not is_valid:
                all_valid = False
    
    if all_valid:
        print("\n" + "="*60)
        print("✓ SUCCESS! All syntax errors fixed!")
        print("You can now run: granite --fips 47065000600 --epochs 30")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Some files still have errors.")
        print("Please check the error messages above and fix manually.")
        print("="*60)


if __name__ == "__main__":
    main()