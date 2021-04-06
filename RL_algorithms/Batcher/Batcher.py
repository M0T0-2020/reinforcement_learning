class Batcher:
    def make_batch(self, replay_memory, batch_size):
        return replay_memory.sample(batch_size)