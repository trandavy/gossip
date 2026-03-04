import json
import random
import time

class HumorEngine:
    def __init__(self, json_path='data/quotes.json', update_interval=4.0):
        """
        Assigns text strings based on Track ID and time state machine
        rather than heavy real-time LLM inferences.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.personalities = data['personalities']
        self.update_interval = update_interval
        
        # tid: {'personality_idx': int, 'quote': str, 'last_update': float}
        self.actor_states = {}
        
        # Global tracker for quote cooldowns: { quote_str: last_used_timestamp }
        self.recent_quotes = {}
        self.cooldown_sec = 30.0

    def update(self, active_actor_ids):
        """
        Takes active actors, removes those out of scope, 
        and updates phrases for those still remaining on a timer.
        """
        now = time.time()
        
        # Cleanup stale state
        current_ids = set(self.actor_states.keys())
        for tid in current_ids - active_actor_ids:
            del self.actor_states[tid]

        # Add or update actors
        for tid in active_actor_ids:
            if tid not in self.actor_states:
                # Deterministic personality assignment
                p_idx = tid % len(self.personalities)
                self.actor_states[tid] = {
                    'personality_idx': p_idx,
                    'quote': self.get_quote_with_cooldown(p_idx, now),
                    'last_update': now
                }
            else:
                # Timer check for next quote
                if now - self.actor_states[tid]['last_update'] > self.update_interval:
                    p_idx = self.actor_states[tid]['personality_idx']
                    self.actor_states[tid] = {
                        'personality_idx': p_idx,
                        'quote': self.get_quote_with_cooldown(p_idx, now),
                        'last_update': now
                    }

        # Return a simple mapping of actor ID to string text
        return {tid: v['quote'] for tid, v in self.actor_states.items()}

    def get_quote_with_cooldown(self, p_idx, now):
        """Fetch a random quote enforcing the 30-second global cooldown."""
        quotes = self.personalities[p_idx]['quotes']
        
        # 1. Group quotes that have cooled down
        available_quotes = []
        for q in quotes:
            if q not in self.recent_quotes or (now - self.recent_quotes[q]) > self.cooldown_sec:
                available_quotes.append(q)
                
        # 2. Fallback: If we somehow exhausted all quotes for this personality, just reset
        if not available_quotes:
            available_quotes = quotes
            
        chosen = random.choice(available_quotes)
        
        # 3. Mark it as used
        self.recent_quotes[chosen] = now
        return chosen
