from dataclasses import dataclass
import os
import fnmatch
from typing import Any, Dict, List


@dataclass
class DirectoryTree:
    tree: Dict[str, dict | None]
    num_files: int
    readme_path: str = None
    readme_str: str = None

    def __post_init__(self):
        pass

    ##### Initialization and I/O #####

    @classmethod
    def from_strs(cls, strs):
        tree = {}
        for s in strs:
            parts = [p for p in os.path.normpath(s).split(os.path.sep) if p]
            _insert_path(tree, parts)

        return cls(tree=tree, num_files=len(set(strs)))
    
    @classmethod
    def from_json(cls, obj):
        return cls(
            tree=obj['tree'],
            num_files=obj['num_files'],
            readme_path=obj['readme_path'],
            readme_str=obj['readme_str'],
        )
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'tree': self.tree,
            'num_files': self.num_files,
            'readme_path': self.readme_path,
            'readme_str': self.readme_str,
        }

    def __str__(self) -> str:
        return '\n'.join(self.get_paths())
    
    ##### Tree operations #####

    def add_paths(self, paths: List[str]):
        n_new_paths = 0
        for s in paths:
            if not self.contains_path(s):
                n_new_paths += n_new_paths
                parts = [p for p in os.path.normpath(s).split(os.path.sep) if p]
                _insert_path(self.tree, parts)
        self.num_files += n_new_paths
    
    def get_paths(self, prefix: str = '', depth: int = None, exclude_readme: bool = False) -> List[str]:
        if self.tree is None:
            return []
        
        tree = self.tree
        if prefix:
            tree = get_subtree(tree=tree, prefix=prefix)
        if depth:
            tree = depth_truncate_tree(tree=tree, depth=depth)
        
        paths = get_paths_list_recurse(tree=tree) 

        if exclude_readme:
            paths = [x for x in paths if x != self.readme_path]

        return paths
    
    def contains_path(self, path: str, must_be_file: bool = False) -> bool:
        if self.tree is None:
            return False
        return contains_path(tree=self.tree, path=path, must_be_file=must_be_file)


def _insert_path(tree, parts):
    if not parts:
        return
    head, *tail = parts
    if head not in tree:
        tree[head] = None if not tail else {}
    if tail:
        _insert_path(tree[head], tail)


def filter_tree(tree: Dict[str, str | None], prefix: str) -> Dict[str, str | None]:
    '''Returns a tree containing only items matching the prefix, preserving the full path structure'''
    
    if not prefix:
        return tree
    
    # Parse the prefix into parts
    parts = [p for p in os.path.normpath(prefix).split('/') if p]
    
    if not parts:
        return tree
    
    # Navigate through the tree, tracking matched keys (not patterns)
    def navigate_and_rebuild(node, part_idx, matched_path):
        """Navigate through tree and rebuild structure with matched keys"""
        if not isinstance(node, dict):
            return {}
        
        if part_idx >= len(parts):
            return node
        
        part = parts[part_idx]
        is_last_part = part_idx == len(parts) - 1
        has_wildcard = '*' in part or '?' in part
        
        # Find matching keys
        if has_wildcard:
            matching_keys = [key for key in node.keys() if fnmatch.fnmatch(key, part)]
        else:
            matching_keys = [part] if part in node else []
        
        if not matching_keys:
            return {}
        
        if is_last_part:
            # Last part: return matched keys with their values
            if has_wildcard:
                result = {}
                for key in matching_keys:
                    result[key] = node[key]
                return result
            else:
                # Exact match at end: navigate into this node
                next_node = node[matching_keys[0]]
                if isinstance(next_node, dict):
                    return next_node
                else:
                    return {matching_keys[0]: next_node}
        else:
            # Not last part
            if has_wildcard:
                # Wildcard in middle: preserve matched keys in structure
                result = {}
                for key in matching_keys:
                    if isinstance(node[key], dict):
                        sub = navigate_and_rebuild(node[key], part_idx + 1, matched_path + [key])
                        if sub:
                            result[key] = sub
                return result
            else:
                # Exact match in middle: continue navigation
                next_node = node[matching_keys[0]]
                return navigate_and_rebuild(next_node, part_idx + 1, matched_path + [matching_keys[0]])
    
    # Get the subtree content
    subtree = navigate_and_rebuild(tree, 0, [])
    
    if not subtree:
        return {}
    
    # Rebuild the structure with non-wildcard parts only
    # For non-wildcard parts, we wrap them; for wildcard parts, the keys are already in subtree
    result = subtree
    for part in reversed(parts):
        # Only wrap if it's not a wildcard pattern
        if '*' not in part and '?' not in part:
            result = {part: result}
    
    return result


def get_subtree(tree: Dict[str, str | None], prefix: str) -> Dict[str, str | None]:
    '''Returns the contents at the prefix location'''

    # If no prefix, return entire tree
    if not prefix:
        return tree
    
    # Parse the prefix into parts
    parts = [p for p in os.path.normpath(prefix).split('/') if p]
    
    if not parts:
        return tree
    
    def navigate_recursive(node, part_idx):
        """Recursively navigate and return matching subtree"""
        if not isinstance(node, dict):
            return {}
        
        if part_idx >= len(parts):
            return node
        
        part = parts[part_idx]
        is_last_part = part_idx == len(parts) - 1
        has_wildcard = '*' in part or '?' in part
        
        # Find matching keys
        if has_wildcard:
            matching_keys = [key for key in node.keys() if fnmatch.fnmatch(key, part)]
        else:
            matching_keys = [part] if part in node else []
        
        if not matching_keys:
            return {}
        
        if is_last_part:
            # Last part
            if has_wildcard:
                # Wildcard at end: return matched keys with their values
                result = {}
                for key in matching_keys:
                    result[key] = node[key]
                return result
            else:
                # Exact path at end: navigate into this node
                next_node = node[matching_keys[0]]
                if isinstance(next_node, dict):
                    return next_node
                else:
                    return {matching_keys[0]: next_node}
        else:
            # Not last part
            if has_wildcard:
                # Wildcard in middle: preserve the wildcard-matched keys in structure
                result = {}
                for key in matching_keys:
                    if isinstance(node[key], dict):
                        sub = navigate_recursive(node[key], part_idx + 1)
                        if sub:
                            result[key] = sub
                return result
            else:
                # Exact match in middle: just navigate
                next_node = node[matching_keys[0]]
                return navigate_recursive(next_node, part_idx + 1)
    
    return navigate_recursive(tree, 0)

def depth_truncate_tree(tree: Dict[str, str | None], depth: int) -> Dict[str, str | None]:
    '''Truncates tree so everything after a certain depth is removed'''
    
    def truncate_recursive(node, current_depth):
        if not isinstance(node, dict):
            return node
        
        # If we've reached the depth limit, return empty dict (no children)
        if current_depth >= depth:
            return {}
        
        result = {}
        for key, value in node.items():
            if current_depth + 1 >= depth:
                # At depth-1, only keep the keys but mark as leaf (None)
                result[key] = None
            else:
                # Recurse deeper
                if isinstance(value, dict):
                    result[key] = truncate_recursive(value, current_depth + 1)
                else:
                    result[key] = value
        
        return result
    
    return truncate_recursive(tree, 0)

def contains_path(tree: Dict[str, dict | None], path: str, must_be_file: bool = False) -> bool:
    parts = [p for p in os.path.normpath(path).split('/') if p]
    node = tree

    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]

    # After traversal:
    if must_be_file:
        return node is None
    return True

def get_paths_list_recurse(tree: Dict[str, str | None], curr_path: str= "") -> List[str]:
    '''Returns tree as an alphabetized list of paths'''

    paths = []
    for key in sorted(tree.keys()):
        value = tree[key]

        new_path = f"{curr_path}/{key}" if curr_path else key
        if isinstance(value, dict):
            paths.extend(get_paths_list_recurse(value, new_path))
        elif value is None:
            # It's a file, so just add the path
            paths.append(new_path)
        else:
            raise Exception(new_path)
    return paths
