import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt 
import tflowtools as tft 


# ***** Artificial Neural Network *****


class Gann():

	def __init__(self, layerSizes, caseman, learningRate = .1, showInterval = None,
				 miniBatchSize = 10, validationInterval = None, softmaxOutputs = False):

		self.learningRate = learningRate
		self.layerSizes = layerSizes			# Size of each layer of neurons
		self.showInterval = showInterval		# Frequency of showing grabbed variables
		self.globalTrainingStep = 0				# Enables coherent data-storage during extra training runs
		self.grabvars = []						# Variables to be monitored during a run.
		self.grabvarFigures = []				# One matplotlib figure for each grabvar.
		self.miniBatchSize = miniBatchSize
		self.validationInterval = validationInterval
		self.validationHistory = []
		self.caseman = caseman
		self.softmaxOutputs = softmaxOutputs
		self.modules = []
		self.build()

	# Probeb variables are to be displayed in the Tensorboard.
	def genProbe(self, moduleIndex, type, spec):
		self.modules[moduleIndex].genProbe(type, spec)

	# Grabvars are displayed by ...
	def addGrabvar(self, moduleIndex, type = 'wgt'):
		self.grabvars.append(self.modules[moduleIndex].getvar(type))
		self.grabvarFigures.append(plt.figure())

	def roundupProbes(self):
		self.probes = tf.summary.merge_all()

	def addModule(self, module):
		self.modules.append(module)

	def build(self):
		tf.reset_default_graph() 	# This is essential for doing multiple runs
		numInputs = self.layerSizes[0]
		self.input = tf.placeholder(tf.float64, shape = (None, numInputs), name = 'Input')
		invar = self.input; insize = num_inputs

		# Build all of the modules
		for i, outsize in enumerate(self.layerSizes[1:]):
			gmod = Gannmodule(self, i, invar, insize, outsize)
			invar = gmod.output; insize = gmod.outsize

		self.output = gmod.output 	# Output of last module is output of whole network

		if self.softmaxOutputs: self.output = tf.nn.softmax(self.output)

		self.target = tf.placeholder(tf.float64, shape = (None, gmod.outsize), name = 'Target')

		self.configureLearning()

	def configureLearning(self):
		self.error = tf.reduce_mean(tf.square(self.target - self.outpost), name = 'MSE')
		self.predictor = self.output 	# Simple prediction runs will request the value of output neurons

		# Defining the training operator
		optimizer = tf.train.GradientDescentOptimizer(self.learningRate)
		self.trainer = optimizer.minimize(self.error, name = 'Backdrop')

	def doTraining(self, sess, cases, epochs = 100, continued = False):
		if not(continued):
			self.errorHistory = []

		for i in range(epochs):
			error = 0; step = self.globalTrainingStep + 1
			gvars = [self.error] + self.grabvars
			mbs = self.miniBatchSize; nCases = len(cases); nmb = math.ceil(nCases/mbs)

			for cstart in range(0, ncases, mbs):
				cend = min(ncases, cstart+mbs)
				minibatch = cases[cstart:cend]
				inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
				feeder = {self.input: inputs, self.target: targets}
				_, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session = sess,
										feed_dict = feeder, step = step, show_interval = self.showInterval)
				error += grabvals[0]

			self.errorHistory.append((step, error/nmb))
			self.considerValidationTesting(step, sess)

		self.globalTrainingStep += epochs
		tft.plot_training_history(self.error_history, self.validation_history, xtitle = "Epoch", 
						ytitle = "Error", title = "", fig = not(continued))


	def doTesting(self, sess, cases, msg = 'Testing', bestk = None):
		inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
		feeder = {self.input: inputs, self.target: targets}
		self.testFunc = self.error

		if bestk is not None:
			self.testFunc = self.genMatchCounter(self.predictor, [tft.one_hot_to_int(list(v)) for v in targets], k = bestk)

		testres, grabvals, _ = self.run_one_step(self.testFunc, self.grabvars, self.probes, session=sess,
								feed_dict = feeder, show_interval = None)

		if bestk is None:
			print('%s Set Error = %f ' % (msg, testres))
		else:
			print('%s Set Correct Classification = %f %%' % (msg, 100*(testres/len(cases))))

		# self.error uses MSE, so this is a per-case value when bestk=None
		return testres

	# Logits = tensor, float - [batch_size, NUM_CLASSES].
	# labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
	# in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
	# This returns a OPERATION object that still needs to be RUN to get a count.
	# tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
	# the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
	# problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
	# target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

	def genMatchCounter(self, logits, labels, k=1):
		correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k) # Return number of correct outputs
		return tf.reduce_sum(tf.cast(correct, tf.int32))

	def trainingSession(self, epochs, sess = None, dir = "probeview", continued = False):
		self.roundupProbes()
		session = sess if sess else tft.gen_initialized_session(dir=dir)
		self.currentSession = session
		self.doTraining(session, self.caseman.get_training_cases(), epochs, continued = continued)

	def testingSession(self, sess, bestk = None):
		cases = self.caseman.get_training_cases()
		if len(cases) > 0:
			self.doTesting(sess, cases, msg = 'Final testing', bestk = bestk)

	def considerValidationTesting(self, epoch, sess):
		if self.validationInterval and (epoch % self.validationInterval == 0):
			cases = self.caseman.get_validation_cases()
			if len(cases) > 0:
				error = self.doTesting(sess, cases, msg = 'Validation Testing')
				self.validation_history.append((epoch, error))

	# Do testing on the training set - calc error without learning
	def testOnTrains(self, sess, bestk = None):
		self.doTesting(sess, self.caseman.get_training_cases(), msg = 'Total Training', bestk = bestk)

	def runOneStep(self, operators, grabbedVars = None, probedVars = None, dir = 'probeview',
				session = None, feedDict = None, step = 1, showInterval = 1):
		sess = session if session else tft.gen_initialized_session(dir=dir)
		if probed_vars is not None:
			results = sess.run([operators, grabbedVars, probedVars], feed_dict = feedDict)
			sess.probe_stream.add_summary(results[2], global_step = step)

		else:
			results = sess.run([operators, grabbedVars, probedVars], feed_dict = feedDict)

		if showInterval and (step % showInterval == 0):
			self.displayGrabvars(results[1], grabbedVars, step = step)

		return results[0], results[1], sess

	def displayGrabvars(self, grabbedVals, grabbedVars, step = 1):
		names = [x.name for x in grabbedVars];
		msg = "Grabbed Variables at Step " + str(step)
		print("\n" + msg, end = "\n")
		figIndex = 0
		for i, v in enumerate(grabbedVars):
			if names: print("   " + names[i] + " = ", end="\n"):
				tft.hinton_plot(v, fig = self.grabvarFigures[figIndex], title = names[i] + ' at step ' + str(step))
				figIndex += 1

			else:
				print(v, end="\n\n")

	def run(self, epochs=100, sess = None, continued = False, bestk = None):
		plt.ion()
		self.trainingSession(epochs, sess = sess, continued = continued)
		self.testOnTrains(sess = self.currentSession, bestk = bestk)
		self.testingSession(sess = self.currentSession, bestk = bestk)
		self.closeCurrentSession(view = False)
		plt.ioff()

	def runmore(self, epochs = 100, bestk = None):
		self.reopenCurrentSession()
		self.run(epochs, sess = self.currentSession, continued = true, bestk = bestk)

	def saveSessionParams(self, spath='netsaver/my_saved_session', sess = None, step = 0):
		session = sess if sess else self.currentSession
		stateVars = []
		for m in self.modules:
			vars = [m.getvar('wgt'), m.getvar('bias')]
			stateVars = stateVars + vars

		self.stateSaver = tf.train.Saver(state_vars)
		self.savedStatePath = self.sate_saver-save(session, spath, global_step = step)


	def reopenCurrentSession(self):
		self.currentSession = tft.copy_session(self.currentSession) # Open a new session with the same tensors
		self.currentSession.run(tf.global_variables_initializer())
		self.restore_session_params() 	# Reload weights and biases


	def restoreSessionParams(self, path = None, sess = None):
		spath = path if path else self.savedStatePath
		session = sess if sess else self.currentSession
		self.stateSaver.restore(session, spath)


	def closeCurrentSession(self, view = True):
		self.saveSessionParams(sess = self.currentSession)
		tft.close_session(self.currentSession, view = view)





class Gannmodule():

	def __init__(self, ann, index, invariable, insize, outsize):
		self.ann = ann
		self.insize = insize
		self.outsize = outsize
		self.input = invariable
		self.index = index
		self.name = "Module-" + str(self.index)
		self.build()

	def build(self):
		mona = self.name; n = self.outsize
		self.weights = tf.Variable(np.random.uniform(-.1, .1, size = (self.insize, n)),
					name = mona + '-wgt', trainable = True)
		self.biases = tf.Variable(np.random.uniform(-.1, .1, size = n), name = mona+'-bias', trainable = True)
		self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name = mona + '-out')
		self.ann.add_module(self)

	def getVar(self, type):
		return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

	def genProbe(self, type, spec):
		var = self.getvar(type)
		base = self.name + '_' + type

		with tf.name_scope('probe_'):
			if ('avg' in spec) or ('stdev' in spec):
				avg = tf.reduce_mean(var)
			if 'avg' in spec:
				tf.summary.scalar(base + '/avg/', avg)
			if 'max' in spec:
				tf.summary.scalar(base + '/max/', tf.reduce_max(var))
			if 'min' in spec:
				tf.summary.scalar(base + '/min/', tf.reduce_min(var))
			if 'hist' in spec:
				tf.summary.histogram(base + '/hist', var)


# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

	def __inti__(self, cfunc, vfrac=0, tfrac = 0):
		self.casefunc = cfunc
		self.validationFraction = vfrac
		self.testFraction = tfrac
		self.trainingFrac = 1 - (vfrac + tfrac)
		self.generateCases()
		self.organizeCases()

	def generateCases():
		self.cases = self.casefunc()  # Run the case generator.

	def organizeCases(self):
		ca = np.array(self.cases)
		np.random.shuffe(ca)
		separator1 = round(len(self.cases) * self.trainingFraction)
		separator2 = separator1 + round(len(self.cases) * self.validationFraction)
		self.trainingCases = ca[0:separator1]
		self.validationCases = ca[separator1:separator2]
		self.testingCases = ca[separator2:]

	def getTrainingCases(self): return self.trainingCases
	def getValidationCases(self): return self.validationCases
	def getTestingCases(self): return self.testingCases


#   ****  MAIN functions ****

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.

def autoex(epochs = 300, nbits = 4, lrate = 0.03, showint = 100, mbs = None, vfrac = 0.1, tfrac = 0.1,
			vint = 100, sm = False, bestk = None):

	size = 2**nbits
	mbs = mbs if mbs else size
	caseGenerator = (lambda: tft.gen_all_one_hot_cases(2**nbits))
	cman = Caseman(cfunc = caseGenerator, vfrac = vfrac, tfrac = tfrac)
	ann = Gann(layerSizes = [size, nbits, size], caseman = cman, learningRate = lrate, showInterval = showint, 
		miniBatchSize = mbs, validationInterval = vint, softmaxOutputs = sm)

	#ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
	#ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
	#ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
	ann.run(epochs,bestk=bestk)
	ann.runmore(epochs*2,bestk=bestk)
	return ann

def countex(epochs=5000,nbits=10,ncases=500,lrate=0.5,showint=500,mbs=20,vfrac=0.1,tfrac=0.1,vint=200,sm=True,bestk=1):
    caseGenerator = (lambda: TFT.gen_vector_count_cases(ncases,nbits))
    cman = Caseman(cfunc=caseGenerator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(layerSizes=[nbits, nbits*3, nbits+1], caseman=cman, learningRate=lrate, showInterval=showint, 
    		miniBatchSize=mbs, validationInterval=vint, softmaxOutputs=sm)
    ann.run(epochs,bestk=bestk)
    return ann










































