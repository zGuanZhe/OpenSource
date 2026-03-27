import time
class Benchmark:
    @staticmethod
    def measure_generation_speed(generator, prompt, max_length=50):
        start_time = time.time()
        output = generator.generate(prompt, max_length=max_length)
        end_time = time.time()
        
        duration = end_time - start_time
        tokens_per_second = max_length / duration
        print(f"Generated {max_length} tokens in {duration:.2f}s ({tokens_per_second:.2f} tokens/s)")
        return output, tokens_per_second
