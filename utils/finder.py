from fnmatch import fnmatch
import os
import re
def walk(root_path, type_filter=None, pattern=None, ignored_patterns=(),
         re_pattern=None, return_basename=False, recursive=False,
         ignore_hidden_directories=True, include_folder_name=False) -> list:
  """Traverse through all files/folders under given `root_path`

  :param root_path: directory to be traversed
  :param type_filter: specify what type of path should be returned, can be
                      (1) None (default), indicating return all paths
                      (2) `file` for filtering out file paths
                      (3) `folder` or `dir` for filtering out directories
  :param pattern: patterns to be included using fnmatch
  :param ignored_patterns: patterns to be ignored using fnmatch
  :param re_pattern: patterns to be included using regular expression
  :param return_basename: whether to return the base name
  :param recursive: whether to traverse recursively.
                    If set to True, `return_basename` must be False
  :param ignore_hidden_directories: whether to ignore hidden directories
  :param include_folder_name: whether to include folder name when `type_filter`
                              is `file`. Designed for `tree` method
  :return: a list of paths
  """
  # Sanity check
  if not os.path.exists(root_path): raise FileNotFoundError(
    '!! Directory `{}` does not exist.'.format(root_path))

  # Get all paths under root_path
  paths = [os.path.join(root_path, p).replace('\\', '/')
           for p in os.listdir(root_path)]

  # Sort paths to avoid the inconsistency between Windows and Linux
  paths = sorted(paths)

  # Define filter function
  def filter_out(f, filter_full_path=False):
    if not callable(f): return paths
    _f = f if filter_full_path else lambda p: f(os.path.basename(p))
    return list(filter(_f, paths))

  # Filter path
  if type_filter in ('file', os.path.isfile):
    type_filter = os.path.isfile
  elif type_filter in ('folder', 'dir', os.path.isdir):
    type_filter = os.path.isdir
  elif type_filter is not None: raise ValueError(
    '!! `type_filter` should be one of (`file`, `folder`, `dir`)')
  paths = filter_out(type_filter, filter_full_path=True)

  # Include pattern using fnmatch if necessary
  if pattern is not None:
    paths = filter_out(lambda p: fnmatch(p, pattern))

  # Ignore patterns using fnmatch if necessary
  if isinstance(ignored_patterns, str): ignored_patterns = ignored_patterns,
  assert isinstance(ignored_patterns, (tuple, list))
  if ignore_hidden_directories:
    ignored_patterns = list(ignored_patterns) + ['.*']
  if len(ignored_patterns) > 0:
    paths = filter_out(
      lambda p: not any([fnmatch(p, ptn) for ptn in ignored_patterns]))

  # Filter pattern using regular expression if necessary
  if re_pattern is not None:
    paths = filter_out(lambda p: re.match(re_pattern, p) is not None)

  # Return if required
  if return_basename:
    if recursive: raise AssertionError(
      '!! Can not return base name when traversing recursively')
    return [os.path.basename(p) for p in paths]

  # Traverse sub-folders if required
  _walk = lambda p, t, r, ptn=None: walk(
    p, type_filter=t, pattern=ptn, re_pattern=re_pattern,
    ignored_patterns=ignored_patterns, return_basename=False, recursive=r,
    include_folder_name=include_folder_name)
  if recursive:
    for p in _walk(root_path, 'dir', False):
      # Walk into `p` first, if no targets found, skip
      sub_paths = _walk(p, type_filter, True, ptn=pattern)
      if len(sub_paths) == 0: continue

      if p in paths:
        paths.remove(p)
        paths.append(p)
      if p not in paths and include_folder_name:
        paths.append(p)
      paths.extend(sub_paths)
  return paths

