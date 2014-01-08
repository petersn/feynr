#! /usr/bin/python

import re
import math

begin_env = re.compile(r"\\begin\s*[{]feynr[}]", re.MULTILINE)
end_env = re.compile(r"\\end\s*[{]feynr[}]", re.MULTILINE)
feynr_options = re.compile(r"\\feynroptions\s*[{]([^}]*)[}]", re.MULTILINE)

anti_particle = {
	"electron": "positron",
	"positron": "electron",
	"photon": "photon",
}

def lerp(a, b, coef):
	return a[0]*(1-coef) + b[0]*coef, a[1]*(1-coef) + b[1]*coef

def error(s):
	print "="*len(s)
	print s
	print "="*len(s)
	exit(1)

def _compute_remaining_angles(angles):
	angles = sorted([angle % (math.pi*2) for angle in angles])
	if len(angles) == 1:
		return angles[0] - 2*math.pi/3, angles[0] + 2*math.pi/3
	elif len(angles) == 2:
		arc12, arc21 = angles[1]-angles[0], (angles[0]+2*math.pi)-angles[1]
		if arc12 != arc21:
			return [(angles[0]+angles[1])/2.0 + math.pi * (arc12 < arc21)]
		else:
			# The two angles are exactly opposite.
			# Simply choose an arbitrary symmetry breaking rule.
			return [angles[0] + math.pi/2.0]
	else: assert False

def compute_remaining_angles(angles):
	r = _compute_remaining_angles(angles)
	r = [i%(math.pi*2) for i in r]
	print "Computing remaining:", [i/math.pi for i in angles], "->", [i/math.pi for i in r]
	return r

def short_way_around(a, b):
	def make_more_than(a, b):
		if b < a: b += 2*math.pi
		return b
	if make_more_than(a, b) - a < make_more_than(b, a) - b:
		return a, make_more_than(a, b)
	else:
		return b, make_more_than(b, a)

def remove_latex_comments(s):
	"""remove_latex_comments(s) -> s, but after a crummy attempt to remove comments"""
	bilge = "RaNdOmNoNcE"
	return "\n".join(i.replace("\\%", bilge).split("%", 1)[0].replace(bilge, "\\%") for i in s.split("\n"))

global_config = {
	"rotate": 0,
	"flip": 0,
	"photon-frequency": 1.0,
	"photon-amplitude": 1.0,
}

direct_options = ["flip", "rotate", "photon-frequency", "photon-amplitude"]

def global_interpret(opt):
	print "="*5, "\\feynroptions{%s}" % opt
	if opt == "time-up":
		global_config["rotate"] = 0
		global_config["flip"] = False
	elif opt == "time-down":
		global_config["rotate"] = 0
		global_config["flip"] = True
	elif opt == "time-left":
		global_config["rotate"] = 270
		global_config["flip"] = True
	elif opt == "time-right":
		global_config["rotate"] = 270
		global_config["flip"] = False
	elif any(opt.startswith(i+"=") for i in direct_options):
		for i in direct_options:
			if opt.startswith(i+"="):
				global_config[i] = float(opt[len(i)+1:])
	else:
		error("Unknown \\feynroptions flag: %s" % opt)

def extract_envs(s):
	s = remove_latex_comments(s)
	# Find all the options passed.
	for opts in feynr_options.findall(s):
		opts = [i.strip() for i in opts.split(",") if i.strip()]
		map(global_interpret, opts)
	envs = []
	while True:
		match = begin_env.search(s)
		if not match: break
		s = s[match.end():]
		match = end_env.search(s)
		if not match:
			error("Unterminated feynr environment!")
		envs.append(s[:match.start()])
		s = s[match.end():]
	return envs

class Node:
	def __init__(self, parent, node, flags):
		self.parent, self.node, self.flags = parent, node, flags
		self.level = None
		self.source_type = None
		for flag in flags:
			if flag == "input":
				parent.inputs.append(self)
			elif flag == "output":
				parent.outputs.append(self)
			elif flag in ("electron", "positron", "photon"):
				self.source_type = flag
			else: assert "Invalid flag:", flag

	def to_code(self):
		return "%% %s\n" % (self.node)

	def logic(self):
		self.lines = set()
		self.resolved = []
		self.unresolved_lines = []
		for line in self.parent.elements:
			if line.obj_a == self or line.obj_b == self and not line.free:
				self.lines.add(line)
				if line.angle is not None:
					self.resolved.append(line.angle_from(self))
				else:
					self.unresolved_lines.append(line)

class Line:
	def __init__(self, parent, a, b, flags=[]):
		self.parent, self.a, self.b, self.flags = parent, a, b, flags
		self.type = self.angle = None
		self.visible = True
		self.free = False
		for flag in flags:
			if flag in ("electron", "positron", "photon"):
				self.type = flag
			elif flag == "hidden":
				self.visible = False
			elif flag == "free":
				self.free = True
			elif flag == "0time":
				self.angle = 0
			elif flag == "0space":
				self.angle = math.pi/2.0
			else: assert "Invalid flag:", flag
		if not self.free:
			self.obj_a, self.obj_b = map(parent.node_map.__getitem__, (self.a, self.b))

	def set_type_from(self, node, t):
#		assert self.type is None, "Line being double-set! Inconsistent propagation."
		# Propagation one way is anti-propagation the other way!
		if node is self.obj_b: t = anti_particle[t]
		self.type = t

	def set_angle_from(self, obj, angle):
		if obj is self.obj_a:
			self.angle = angle
		elif obj is self.obj_b:
			self.angle = angle + math.pi
		else: assert False

	def angle_from(self, obj):
		return self.angle + math.pi * (obj == self.obj_b)

	def not_me(self, obj):
		return self.obj_b if obj is self.obj_a else self.obj_a

	def xys(self):
		if not self.free:
			return self.obj_a.xy, self.obj_b.xy
		else:
			# Otherwise special strings may be okay.
			return map(self.parent.get_position, (self.a, self.b))

	def to_code(self):
		if not self.visible:
			return ""
		a_xy, b_xy = self.xys()
		if self.type is None:
			color = "red"
		elif self.type in ("electron", "positron", "photon"):
			color = "black"
		else:
			color = "blue"
		# We have a special case for arcs because otherwise our derivative computation gets screwed up.
		# In the case of arcs we will manually flip things later.
		if self.type == "positron" and not "arc" in self.flags:
			a_xy, b_xy = b_xy, a_xy
		avg = lerp(a_xy, b_xy, 0.54)

		# If we're an arc, compute mid_point and r
		if "arc" in self.flags:
			deriv_a = self.parent.get_deriv(self.a)
			tangent = -deriv_a[1], deriv_a[0] # Rotate CCW by math.pi/2.0
			# Find the x and y components of b_xy
			px = (b_xy[0]*deriv_a[0] + b_xy[1]*deriv_a[1]) - (a_xy[0]*deriv_a[0] + a_xy[1]*deriv_a[1])
			if px < 0:
				deriv_a = -deriv_a[0], -deriv_a[1]
				px *= -1
			py = (b_xy[0]*tangent[0] + b_xy[1]*tangent[1]) - (a_xy[0]*tangent[0] + a_xy[1]*tangent[1])
			if py < 0:
				tangent = -tangent[0], -tangent[1]
				py *+ -1
			# Find the arc that hits a_xy and b_xy while matching deriv_a.
			r = (px**2 + py**2)/(2*px)
			mid_point = a_xy[0]+r*deriv_a[0], a_xy[1]+r*deriv_a[1]
			angle_a = math.atan2(a_xy[1]-mid_point[1], a_xy[0]-mid_point[0])
			angle_b = math.atan2(b_xy[1]-mid_point[1], b_xy[0]-mid_point[0])
			angle_a, angle_b = short_way_around(angle_a, angle_b)
			if "flip" in self.flags:
				angle_a, angle_b = angle_b, angle_a + 2*math.pi
			points = []
			derivs = []
			angles = []
			#for start, stop in ((angle_a, (angle_a+angle_b)/2.0), ((angle_a+angle_b)/2.0, angle_b)):
			angle = angle_a
			while angle < angle_b:
				angles.append(angle)
				angle += 0.1
			angles.append(angle_b)
			for angle in angles:
				points.append((mid_point[0] + math.cos(angle)*r, mid_point[1] + math.sin(angle)*r))
				derivs.append((-math.sin(angle), math.cos(angle)))
			points_str = " ".join("(%f,%f)" % i for i in points)

		if self.type in ("electron", "positron"):
			if "arc" in self.flags:
				if "no-arrow" not in self.flags:
					arrow_angle = (angle_a+angle_b)/2.0
					arrow_pos1 = mid_point[0] + math.cos(arrow_angle)*r, mid_point[1] + math.sin(arrow_angle)*r
					arrow_pos2 = mid_point[0] + math.cos(arrow_angle+1e-3)*r, mid_point[1] + math.sin(arrow_angle+1e-3)*r
					return """\draw[%s, ->] (%f,%f) -- (%f,%f);
\draw[%s] plot[smooth] coordinates {%s};\n""" % ((color,) + arrow_pos1 + arrow_pos2 + (color, points_str))
				else:
					return "\draw[%s] plot[smooth] coordinates {%s};" % (color, points_str)
			else:
				if "no-arrow" not in self.flags:
					return """\draw[%s, ->] (%f,%f) -- (%f,%f);
\draw[%s] (%f,%f) -- (%f,%f);\n""" % ((color,) + a_xy + avg + (color,) + avg + b_xy)
				else:
					return "\draw[%s] (%f,%f) -- (%f,%f);\n" % ((color,) + a_xy + b_xy)
		else:
			if "arc" in self.flags:
				distance = r * (angle_b - angle_a)
			else:
				distance = ((a_xy[0]-b_xy[0])**2+(a_xy[1]-b_xy[1])**2)**0.5
			wiggle_amp = 0.05*global_config["photon-amplitude"]
			complete_waves = int(round(4*distance*global_config["photon-frequency"]))
			# Draw a photonic sine wave.
			wiggles = complete_waves*8+1
			orthog = b_xy[1]-a_xy[1], a_xy[0]-b_xy[0]
			orthog = lerp((0,0), orthog, -wiggle_amp/(orthog[0]**2 + orthog[1]**2)**0.5)
			if "arc" in self.flags:
				points, orthog_list = [], []
				for i in xrange(wiggles):
					angle = angle_a + (angle_b - angle_a) * (i/float(wiggles-1))
					points.append((mid_point[0] + math.cos(angle)*r, mid_point[1] + math.sin(angle)*r))
					orthog_list.append((wiggle_amp*math.cos(angle), wiggle_amp*math.sin(angle)))
			else:
				points = [lerp(a_xy, b_xy, i/float(wiggles-1)) for i in xrange(wiggles)]
			# Add the little wiggles
			for ind in xrange(len(points)):
				coef = [math.sin(i*(math.pi*2)/8.0) for i in xrange(8)][ind%8]
				if "arc" in self.flags:
					orthog = orthog_list[ind]
				points[ind] = points[ind][0] + orthog[0]*coef, points[ind][1] + orthog[1]*coef
			points_str = " ".join("(%f,%f)" % i for i in points)
#			points_str = points_str % (("sin", "cos")*wiggles)[:wiggles-1]
			return "\draw[%s] plot[smooth] coordinates {%s};\n" % (color, points_str)

class Drawing:
	def __init__(self):
		self.remapping = {}
		self.inputs = []
		self.outputs = []
		self.elements = []
		self.draw_only_elements = []
		self.nodes = []
		self.node_map = {}
		self.accumulated_flags = {}
		self.handled_outputs = set()

	def get_node(self, node):
#		if node.endswith("~1"):
#			return node[:-2]
		return self.remapping.get(node, node)

	def accum_flags(self, node, flags=[], line_flags=[]):
		if node not in self.accumulated_flags:
			self.accumulated_flags[node] = (set(), set())
		t = self.accumulated_flags[node]
		t[0].update(flags)
		t[1].update(line_flags)

	def get_position(self, spec):
		if spec in self.node_map:
			return self.node_map[spec].xy
		elif ":" in spec:
			if spec.count(":") == 2:
				alpha, a, b = spec.split(":", 2)
			elif spec.count(":") == 1:
				a, b = spec.split(":", 1)
				alpha = 0.5
			else: assert False
			r = lerp(self.get_position(a), self.get_position(b), float(alpha))
			print "Lerped result:", r
			return r
		else: assert False, "invalid position spec: %s" % spec

	def get_deriv(self, spec):
		if ":" in spec:
			a, b = spec.split(":")[-2:]
			a, b = map(self.get_position, (a, b))
			v = b[0]-a[0], b[1]-a[1]
			norm = (v[0]**2+v[1]**2)**0.5
			return v[0]/norm, v[1]/norm
		else: assert False, "specs must be between two nodes to arc from"

	def new_node(self, node, flags=[], line_flags=[]):
		if node in self.accumulated_flags:
			print "Collecting flags", self.accumulated_flags[node], "for", node
			t = self.accumulated_flags.pop(node)
			# XXX: Mutation of += renders it non-equivalent. Do not change next two lines.
			flags = flags + list(t[0])
			line_flags = line_flags + list(t[1])
		if "~" in node:
			print "Foobar."
#			assert False
			return node
		curr_node = self.get_node(node)
		if "~" in curr_node:
			base_curr_node, num = curr_node.rsplit("~", 1)
			num = int(num)
			next_node = "%s~%i" % (base_curr_node, num+1)
			print "Placing implicit on renaming.", next_node
			self.place_node(next_node, flags=flags)
			# Add an implicit line.
			self.interact(curr_node, next_node, flags=["no-new"]+line_flags)
		else:
			assert curr_node == node, "remapping should only map numbers"
			next_node = curr_node+"~1"
			print "Placing initial implicit.", next_node
			self.place_node(next_node, flags=flags)
		self.remapping[node] = next_node
		return next_node

	def to_code(self):
		s = "\\begin{tikzpicture}[rotate=%i,xscale=1,yscale=%i]\n" % (global_config["rotate"], -1 if global_config["flip"] else 1)
		for l in (self.elements, self.draw_only_elements):
			s += "\n".join(e.to_code() for e in l)
		s += "\\end{tikzpicture}\n"
		return s

	def interact(self, a, b, flags=[]):
		if ("no-new" not in flags) and ("free" not in flags):
			a, b = map(self.new_node, (a, b))
		if "free" not in flags:
			self.elements.append(Line(self, a, b, flags))
		else:
			self.draw_only_elements.append(Line(self, a, b, flags))

	def place_node(self, node, flags=[]):
#		if "no-new" not in flags:
#			node = self.new_node(node, flags)
		obj = self.node_map[node] = Node(self, node, flags)
		self.nodes.append(obj)

	def get_layers_from(self, start):
		layer = dict((i, 0) for i in start)
		# Inefficient O(n^2), rather than Bellman-Ford, or some such. Oh well.
		inf = float("inf")
		for i in xrange(len(self.nodes)):
			for link in self.elements:
				if link.obj_a in layer:
					layer[link.obj_b] = min(layer.get(link.obj_b, inf), layer[link.obj_a]+1)
				if link.obj_b in layer:
					layer[link.obj_a] = min(layer.get(link.obj_a, inf), layer[link.obj_b]+1)
		return layer

	def type_inference(self):
#		print "Node types:"
#		for node in self.nodes:
#			print node.node, node.source_type
#		print "Line types:"
#		for line in self.elements:
#			print line.a, line.b, "--", line.type
		# Inefficient O(n^2).
		for i in xrange(len(self.nodes)):
			for node in self.nodes:
				if len(node.lines) == 1 and node.source_type is not None:
					tuple(node.lines)[0].set_type_from(node, node.source_type)
				if len(node.lines) == 3:
					# Examine the incoming lines.
					type_counts = {None: 0, "electron": 0, "positron": 0, "photon": 0}
					for line in node.lines:
						type_counts[line.type] += 1
					if type_counts[None] == 1:
						fill_in = anti_particle[[k for k, v in type_counts.items() if v == 0 and k is not None][0]]
						for line in node.lines:
							if line.type is None:
								line.set_type_from(node, fill_in)

	def logic(self):
		# Make nodes precompute stuff.
		for node in self.nodes: node.logic()
		# Compute how far nodes are from the inputs and outputs of our diagram.
		layer_lists = map(self.get_layers_from, (self.inputs, self.outputs))
		for k, v in layer_lists[0].items():
			print k.node, "--", v
		# Start by printing all the initial things.
		for line in self.elements:
			print line.a, line.b, "--", line.angle
		# Next, begin iterating until we have some stasis.
		while True:
			print "Solving pass."
			for node in self.nodes:
				# Only check the nodes with three links.
				if len(node.lines) != 3: continue
				dirty = set((node,))
				if len(node.resolved) in (1, 2):
					print "Resolving", node.node
					new_angles = compute_remaining_angles(node.resolved)
					assert len(new_angles) == len(node.unresolved_lines)
					dirty |= set(i.not_me(node) for i in node.unresolved_lines)
					# Figure out a way to map new_angles onto the unresolved nodes.
					if len(new_angles) == 1:
						node.unresolved_lines[0].set_angle_from(node, new_angles[0])
					elif len(new_angles) == 2:
						# Figure out which angle is more downwardsy
						new_angles = sorted(new_angles, key=math.sin)
						# Figure out which line points more downwardsy
						def line_cmp(a, b):
							a, b = a.not_me(node), b.not_me(node)
							return cmp(layer_lists[0][a], layer_lists[0][b])
						unresolved_lines = sorted(node.unresolved_lines, cmp=line_cmp)
						for i in (0, 1): unresolved_lines[i].set_angle_from(node, new_angles[i])
					else: assert False
					for node in dirty: node.logic()
					break
			else: break
		# See if there are any unresolved lines.
#		still_unresolved = [(line.a, line.b) for line in self.elements if line.angle is None]
#		if still_unresolved:
#			print "STILL UNRESOLVED LINES:"
#			for line in still_unresolved:
#				print line
#			exit(1)
		# Propagate positions.
		already_hit = set()
		def traverse(node):
			if node in already_hit: return
			already_hit.add(node)
			for line in node.lines:
				end = line.not_me(node)
				if line.angle is not None:
					length = 1.5 if "long" in line.flags else 1.0
					end.xy = node.xy[0] + math.cos(line.angle_from(node))*length, node.xy[1] + math.sin(line.angle_from(node))*length
					traverse(end)
#			for line in node.lines:
#				traverse(line.not_me(node))
		self.inputs[0].xy = (0, 0)
		traverse(self.inputs[0])
		# Finally, do type inference to figure out what type each link is.
		self.type_inference()
		for i in self.nodes:
			if hasattr(i, "xy"): print i.node, i.xy
			else: print i.node

def get_flags(l):
	return l[l.index("--")+1:] if "--" in l else []

def process_macros(code):
	output = []
	for original_line in code:
		line = original_line.split("%")[0].strip()
		if not line: continue
		line = line.split(" ")
		cmd, args = line[0], line[1:]
		if cmd == "photon-loop":
			assert len(args) == 2
			for line in """draw {a} 0.3:{a}:{b} photon
draw 0.3:{a}:{b} 0.7:{a}:{b} electron arc
draw 0.7:{a}:{b} 0.3:{a}:{b} electron arc
draw 0.7:{a}:{b} {b} photon""".format(a=args[0], b=args[1]).split("\n"):
				output.append(line)
		else:
			output.append(original_line)
	return output

def compile_feynr_code(code):
	d = Drawing()
	code = code.split("\n")
	code = process_macros(code)
	print code
	for line in code:
		line = line.split("%")[0].strip()
		if not line: continue
		line = line.split(" ")
		cmd, args = line[0], line[1:]
		if cmd in ("electron", "positron", "photon"):
			for arg in args:
				d.new_node(arg, flags=[cmd])
		elif cmd in ("-", "interact", "draw", "skeleton"):
			extra = []
			if cmd == "skeleton": extra += ["hidden"]
			if cmd == "draw": extra += ["free"]
			d.interact(args[0], args[1], flags=args[2:]+extra)
		elif cmd in ("input", "output", "propagate"):
			flags = get_flags(args)
			for arg in args:
				if arg == "--": break
				if cmd in ("input", "output"):
					if cmd == "output":
						d.handled_outputs.add(arg)
					d.new_node(arg, flags=[cmd]+flags)
				elif cmd == "propagate":
					d.accum_flags(arg, line_flags=flags)
		else:
			error("Invalid command: %s" % cmd)
	# Finally, repropagate all outputs.
	for output in d.remapping:
		if output not in d.handled_outputs:
			d.new_node(output, flags=["output"])
	d.logic()
	return d.to_code()

if __name__ == "__main__":
	import sys
	if len(sys.argv) != 2:
		print "Usage: feynr_render.py input.tex"
		exit(2)
	fd = open(sys.argv[1])
	data = fd.read()
	fd.close()
	envs = extract_envs(data)
	for i, s in enumerate(envs):
		code = compile_feynr_code(s)
#		print code
		fd = open("feynr%i.tex" % i, "w")
		fd.write(code)
		fd.close()

