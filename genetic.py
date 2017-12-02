from collections import defaultdict, namedtuple
import random
from pprint import pprint
from copy import deepcopy
from math import floor

Input = namedtuple('Input', ['state', 'read'])
Result = namedtuple('Result', ['state', 'write', 'direction'])
directions = [-1, 0, 1]


def to_tape(data, blank):
    tape = [blank, blank, *list(data), blank, blank]
    return tape


class Individual:
    def __init__(self, rules=None, num_inputs=None, num_outputs=None):
        if rules is not None:
            self.rules = rules
        else:
            assert num_inputs and num_outputs
            self.rules = random.sample(range(num_outputs), num_inputs)
        self.num_inputs = num_inputs or len(self.rules)
        self.num_outputs = num_outputs or max(self.rules)

    def __sub__(self, other):
        diff_vector = []
        for rule, other_rule in zip(self.rules, other.rules):
            diff_vector.append(rule - other_rule)
        return diff_vector


    def __iter__(self):
        return (rule for rule in self.rules)



    def add_noise(self, diffs):
        return Individual(rules=[floor(original + diff)
                                 if floor(original + diff) > 0
                                 else 0
                                 for original, diff
                                 in zip(self.rules, diffs)])


class TransitionTable:
    def __init__(self):
        self._table = {}

    def add_rule(self, input_rule, result):
        self._table[input_rule] = result

    def evaluate(self, input_rule):
        try:
            return self._table[input_rule]
        except:
            return None

    def __str__(self):
        return '{!r}'.format(self._table)


class Turing:
    def __init__(self, num_states, alphabet, symbols, transition, blank):
        self.states = ['q{}'.format(i) for i in range(num_states)]
        self.alphabet = alphabet
        self.symbols = symbols
        self.transition = transition
        self.initial_state = self.states[0]
        self.blank = blank
        self.final_states = [self.states[-1]]
        # Classical Optimization Encoding
        self.input_list = [Input(state, symbol)
                           for state in self.states
                           for symbol in self.alphabet]
        self.output_list = [Result(state, symbol, direction)
                            for state in self.states
                            for symbol in self.symbols
                            for direction in (0, 1, 2)]
        self.num_inputs = len(self.input_list)
        self.num_outputs = len(self.output_list)
        # Differential Evolution
        self.dimension = (len(self.states) - 1) * len(self.symbols)

    def predict(self, input_str):
        state = self.initial_state
        head = 2
        tape = to_tape(input_str, self.blank)
        TTL = 1000
        while state not in self.final_states:
            if TTL == 0:
                return None
            if head >= len(tape):
                tape.append(self.blank)
            elif head < 0:
                tape.insert(0, self.blank)
                head = 0
            input_symbol = tape[head]
            result = self.transition.evaluate(Input(state, input_symbol))
            if result is None:
                break
            state = result.state
            tape[head] = result.write
            head += directions[result.direction]
            TTL -= 1
        return ''.join(tape)[2:-2]

    def evaluate(self, input_str, output_str):
        predicted = self.predict(input_str)
        if predicted is None:
            return 10000
        cost_value = 0
        value = 20
        print(predicted, output_str)
        for predicted_value, real in zip(predicted, output_str):
            if predicted_value == real:
                cost_value -= value
            else:
                cost_value += value
        return cost_value

    def from_individual(self, individual):
        self.transition = TransitionTable()
        for input_idx, output_idx in enumerate(individual.rules):
            if output_idx >= self.num_outputs:
                output_idx = self.num_outputs - 1
            self.transition.add_rule(self.input_list[input_idx],
                                     self.output_list[output_idx])

class ClassicalOptimizer:
    def __init__(self, turing, NP=50, G=100, F=0.8, CR=0.6, V=20, CV=0):
        self.turing = turing
        self.NP = NP
        self.G = G
        self.F = F
        self.CR = CR
        self.V = V
        self.CV = CV
        self.input = None
        self.output = None

    def generate_population(self):
        population = []
        for i in range(self.NP):
            population.append(Individual(num_inputs=self.turing.num_inputs,
                                         num_outputs=self.turing.num_outputs))
        return population

    def parse_input(self, description):
        self.input, self.output = description.strip().split(' ')

    def cost_fn(self, individual):
        self.turing.from_individual(individual)
        return self.turing.evaluate(self.input, self.output)

    def evolve_generation(self, population):
        assert len(population) >= 4
        new_population = []
        for individual in population:
            surrogates = []
            for i in range(3):
                surrogate = random.choice(population)
                # If we get a dupe, pick again
                while (surrogate == individual or
                       surrogate in surrogates):
                    surrogate = random.choice(population)
                surrogates.append(surrogate)
            diff_vector = surrogates[0] - surrogates[1]
            weighted_vector = [diff *  self.F for diff in diff_vector]
            noise_individual = surrogates[2].add_noise(weighted_vector)
            test_individual = Individual([test if random.random() < self.CR else original
                                          for original, test
                                          in zip(individual, noise_individual)])
            survivor = min(individual, test_individual, key=self.cost_fn)
            new_population.append(survivor)
        return new_population

    def optimize(self):
        population = self.generate_population()
        for i in range(self.G):
            print()
            new_generation = self.evolve_generation(population)
            population = new_generation
            best_so_far = min(self.cost_fn(indv) for indv in population)
            print('Generation: {} Best: {}'.format(i, best_so_far))
        best_individual = min(population, key=self.cost_fn)
        turing = deepcopy(self.turing)
        turing.from_individual(best_individual)
        return turing


if __name__ == '__main__':
    # Define Turing Machine
    num_states = 2
    symbols = ['0', '1', 'X']
    blank = '#'
    alphabet = ['0', '1', blank]
    transition = TransitionTable()
    turing = Turing(num_states, alphabet, symbols, transition, blank,)
    optimizer = ClassicalOptimizer(turing, NP=10*turing.dimension)
    optimizer.parse_input('1000 1001')
    model = optimizer.optimize()
    print(model.predict('110'))
