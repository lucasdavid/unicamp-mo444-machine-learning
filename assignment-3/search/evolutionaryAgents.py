import json
import os

import numpy as np
import pandas as pd
from game import Agent, Directions
from util import manhattanDistance

MAX_SCARED_TIME = 40

INITIAL_POPULATION = 200
INITIAL_FEATURE_STRENGTH = 2.0
MUTATION = 0.05
MUTATION_STRENGTH = 0.5
TESTS_PER_AGENT = 5
RETEST_EVERY_GENERATION = False
VERBOSE = True
MODEL_SAVED_AT = 'checkpoints/ev/epochs-5000-pop-100-f-str-2-mut-0.05-mut-str-0.5-tests-5-selection-before/'

np.random.seed(153)


class History(object):
  def __init__(self):
    self.data = {}

  def update(self, key, value):
    if key not in self.data:
      self.data[key] = []
    self.data[key].append(value)

  def __getitem__(self, item):
    return self.data[item]

  def items(self):
    return self.data.items()

  def as_pandas(self):
    return pd.DataFrame(self.data)


class EvolutionPool(object):
  def __init__(
      self,
      shape,
      samples=INITIAL_POPULATION,
  ):
    self.shape = shape
    self.samples = samples
    self.individuals = None
    self.scores = None
    self.steps = None
    self.evaluations = None
    self.total_evaluations = 0
    self.generations = 0
    self.current_ind = 0
    self.fittest_idx = -1
    self.fittest_score = -np.inf
    self.fittest_weights = None

    self.history = History()
    self.generate()

  def generate(self):
    if VERBOSE: print('[EvolutionPool] generating an initial population of %i' % self.samples)
    self.individuals = (2 * (np.random.rand(self.samples, *self.shape) - 0.5) * INITIAL_FEATURE_STRENGTH)
    self.scores = np.full(self.samples, -np.inf)
    self.steps = np.zeros(self.samples)
    self.evaluations = dict()

  def training(self, epoch):
    if self.current_ind >= len(self.individuals):
      self.track_generation(epoch)
      self.selection()
      self.reproduce()

    return self.current_ind, self.individuals[self.current_ind]

  def evaluate(self, ind, score, steps):
    if ind not in self.evaluations:  # First eval
      self.evaluations[ind] = []

    self.evaluations[ind].append(score)
    self.steps[ind] += steps
    self.total_evaluations += 1

    if len(self.evaluations[ind]) >= TESTS_PER_AGENT:
      self.scores[ind] = np.mean(self.evaluations[ind])
      self.steps[ind] /= TESTS_PER_AGENT

      if self.scores[ind] > self.fittest_score:
        if VERBOSE: print('[EvolutionPool] fittest candidate found [score=%s]' % self.scores[ind])
        self.fittest_idx = ind
        self.fittest_score = self.scores[ind]
        self.fittest_weights = self.individuals[ind]

      # No more tests for this individual.
      self.current_ind += 1

    return self.scores[ind]

  def track_generation(self, epoch):
    if VERBOSE: print('[EvolutionPool] Epoch=%i' % epoch)

    metrics = (
      ('generation', self.generations),
      ('scores', self.scores.copy()),
      ('scores_max', self.scores.max()),
      ('scores_min', self.scores.min()),
      ('scores_avg', self.scores.mean()),
      ('steps_max', self.steps.max()),
      ('steps_min', self.steps.min()),
      ('steps_avg', self.steps.mean()),
    )

    for k, v in metrics:
      self.history.update(k, v)
      if isinstance(v, np.ndarray): v = v.round()[:5]
      if VERBOSE: print('   %s=%s' % (k, v))

  def reproduce(self):
    old_population_count = len(self.individuals)

    a, b = self.roulette(self.individuals)
    offspring = self.crossover(a, b)
    offspring = self.mutate(offspring)
    self.individuals = np.concatenate((self.individuals, offspring), axis=0)
    self.scores = np.concatenate((self.scores, np.full(len(offspring), -np.inf)))
    self.steps = np.concatenate((self.steps, np.full(len(offspring), 0)))

    self.current_ind = 0 if RETEST_EVERY_GENERATION else old_population_count
    self.generations += 1

  def roulette(self, individuals):
    p = self.scores - self.scores.min() + 1e-7
    pairs = np.random.choice(len(individuals), size=2 * len(individuals), p=p / p.sum())

    a, b = pairs[:len(pairs) // 2], pairs[len(pairs) // 2:]
    return individuals[a], individuals[b]

  def crossover(self, a, b):
    s = np.random.rand(*a.shape) > 0.5
    return s * a + ~s * b

  def mutate(self, individuals):
    r = np.random.rand(*individuals.shape) <= MUTATION
    m = 2 * (np.random.rand(*individuals.shape) - 0.5) * MUTATION_STRENGTH

    return individuals + r * m

  def selection(self):
    samples = len(self.scores) // 2
    ids = np.argsort(self.scores)[::-1]
    selected = ids[:samples]
    self.individuals = self.individuals[selected]
    self.scores = self.scores[selected]
    self.steps = self.steps[selected]
    self.evaluations = {ind: score for ind, score in enumerate(self.scores)}


class EvolutionaryAgent(Agent):
  ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
  FEATURES = [
    'food_h', 'food_v', 'capsule_h', 'capsule_v',
    'scary_ghost_h', 'scary_ghost_v', 'scared_ghost_h', 'scared_ghost_v',
    'legal_north?', 'legal_south?', 'legal_east?', 'legal_west?',
    'previous_north?', 'previous_south?', 'previous_east?', 'previous_west?',
    'bias',
  ]

  def __init__(
      self,
      numTraining=None,
      **kwargs
  ):
    self.epoch = 1
    self.steps = 0
    self.numTraining = numTraining
    self.previous_action = Directions.EAST

    if numTraining:
      if VERBOSE: print('[EvolutionaryAgent] %s epochs will be used for training.' % numTraining)

      self.p = EvolutionPool(
        shape=(len(self.FEATURES), len(self.ACTIONS))
      )

      self.ind, self.w = self.p.training(self.epoch)
    else:
      self.p = None
      self.ind, self.w = -1, load_model(MODEL_SAVED_AT)

  def getAction(self, state):
    previ_actions_mask = np.isin(self.ACTIONS, [self.previous_action])
    legal_actions_mask = np.isin(self.ACTIONS, state.getLegalPacmanActions())

    x = extract_features_from(state, previ_actions_mask, legal_actions_mask)
    y = np.matmul(x, self.w)
    y = np.exp(y) / sum(np.exp(y))
    y = y * legal_actions_mask

    # choice = np.random.choice(self.ACTIONS, p=y / y.sum())
    choice = self.ACTIONS[y.argmax()]

    self.previous_action = choice
    self.steps += 1

    return choice

  def final(self, state):
    if self.epoch < self.numTraining:
      self.p.evaluate(self.ind, state.getScore(), self.steps)
      self.ind, self.w = self.p.training(self.epoch)

    if self.epoch == self.numTraining and self.p.fittest_weights is not None:
      if VERBOSE: print('Restoring best weights [score=%.0f]' % self.p.fittest_score)

      self.w = self.p.fittest_weights
      save_model(self, MODEL_SAVED_AT)

    self.epoch += 1
    self.steps = 0


# region features

def extract_features_from(state, previous_actions, legal_actions):
  pp = state.getPacmanPosition()

  fm = np.asarray(state.getFood().data)
  fp = np.stack(np.where(fm)).T

  cols, rows = fm.shape
  cols, rows = cols-2, rows-2  # account for external walls.

  food, f_idx = distance_as_features(pp, fp, rows, cols)
  capsules, c_idx = distance_as_features(pp, state.getCapsules(), rows, cols)

  ghosts = state.getGhostStates()
  scary = [g for g in ghosts if g.scaredTimer <= 0]
  scared = [g for g in ghosts if g.scaredTimer > 0]

  scary_dist, _ = distance_as_features(pp, [g.getPosition() for g in scary], rows, cols)
  scared_dist, _ = distance_as_features(pp, [g.getPosition() for g in scared], rows, cols)

  # ghost, g_idx = distance_as_features(pp, state.getGhostPositions(), rows, cols)
  # scared = state.getGhostState(g_idx + 1).scaredTimer if g_idx > -1 else 0
  # scared = 1 - scared / MAX_SCARED_TIME

  return np.asarray(food + capsules + scary_dist + scared_dist
                    + legal_actions.tolist()
                    + previous_actions.tolist()
                    + [1])


def distance_as_features(pp, p, rows, cols):
  p = np.asarray(p)
  if not len(p):
    return [1, 1], -1  # max distance

  dists = manhattanDistance(p.T, pp)
  _idx = closest_idx(dists)
  d = p[_idx]
  v = (d - pp).astype('float') / [cols, rows]

  return v.tolist(), _idx


def closest_idx(d, deterministic=False):
  if deterministic:
    return d.argmin()

  indices, = np.where(d == d.min())
  return np.random.choice(indices)

# endregion


# region callbacks

def save_model(model, model_path):
  if VERBOSE: print('[EvolutionaryAgent] saving weights to %s' % model_path)

  try: os.makedirs(model_path)
  except: pass

  with open(os.path.join(model_path, 'weights.txt'), 'w') as f:
    json.dump(model.w.tolist(), f)

  model.p.history.as_pandas().to_csv(os.path.join(model_path, 'history.csv'), index=False)


def load_model(model_path):
  if VERBOSE: print('[EvolutionaryAgent] loading weights from %s' % model_path)

  with open(os.path.join(model_path, 'weights.txt')) as f:
    return np.asarray(json.load(f))

# endregion
