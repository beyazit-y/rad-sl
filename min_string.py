from collections import deque

def compute_min_dist_string(dfa_l, dfa_r):
    # note: the minimum distinguishing string is technically a list of alphabet symbols
    min_dist_str = []
    dfa_l = dfa_l.minimize()
    dfa_r = dfa_r.minimize()
    visited = set()
    if dfa_l != dfa_r:
        queue = deque([(dfa_l, dfa_r, [])])
        visited.add((dfa_l.transition(""), dfa_r.transition(""))) # product state
        while queue:
            curr_dfa_l, curr_dfa_r, curr_word = queue.popleft()
            # if current left DFA is trivially accepting and current right DFA is trivially 
            # rejecting or vice versa, then minimum disinguishing string is found -> break
            if ((curr_dfa_l.label([]) and not curr_dfa_r.label([])) or
                (not curr_dfa_l.label([]) and curr_dfa_r.label([]))):
                min_dist_str = curr_word
                break
            for a in dfa_l.inputs: # both DFAs share the same alphabet
                if (curr_dfa_l.transition([a]), curr_dfa_r.transition([a])) not in visited:
                    queue.append((curr_dfa_l.advance([a]), curr_dfa_r.advance([a]), curr_word + [a]))
                    visited.add((curr_dfa_l.transition([a]), curr_dfa_r.transition([a])))
    return min_dist_str