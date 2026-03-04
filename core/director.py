import time

class Director:
    def __init__(self, max_actors=3, history_thresh=10, cooldown_sec=2.0):
        """
        The Director filters raw tracking data to select the 'Actors' 
        that will have text bubbles displayed over them.
        """
        self.max_actors = max_actors
        self.history_thresh = history_thresh
        self.cooldown_sec = cooldown_sec
        
        self.actors = set()  # IDs of currently active actors
        self.last_removal_time = 0

    def rank_and_select(self, tracks):
        """
        Input: dictionary of currently tracked objects.
        Output: set of track IDs designated as active actors.
        """
        current_track_ids = set(tracks.keys())
        
        # 1. Purge actors who are no longer being tracked
        removed = self.actors - current_track_ids
        if removed:
            self.last_removal_time = time.time()
            self.actors -= removed

        # 2. Promote candidates to actors if space allows and cooldown passed
        if len(self.actors) < self.max_actors and (time.time() - self.last_removal_time > self.cooldown_sec):
            candidates = []
            
            for tid, tdata in tracks.items():
                if tid not in self.actors and tdata['history'] >= self.history_thresh and tdata['lost'] == 0:
                    candidates.append(tid)
            
            # If candidates exist, promote the most stable one to an actor
            if candidates:
                best_candidate = sorted(candidates, key=lambda x: tracks[x]['history'], reverse=True)[0]
                self.actors.add(best_candidate)
                
        return self.actors
