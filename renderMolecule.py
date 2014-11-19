from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from atomicParam import *
import sys
import numpy as np
import time
name = "Molecule"
rotx = 0
roty = 0
posX = 0
posY = 0
rotate = 0


class renderMolecule():
	def __init__(self,cartArray,mol):
		self.CA = cartArray
		self.AT = mol.atomType
		self.BM = mol.bondMatrix
		self.render()
		
	def render(self):
		glutInit(sys.argv)
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
		glutInitWindowSize(400,400)
		glutCreateWindow(name)
		glClearColor(0.,0.,0.,1.)
		glShadeModel(GL_SMOOTH)
		glEnable(GL_CULL_FACE)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)
		lightZeroPosition = [10.,4.,10.,1.]
		lightZeroColor = [0.8,1.0,0.8,1.0] #green tinged
		glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
		glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
		glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
		glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
		gluLookAt(0,0,10, # Camera Position
		          0,0,0,  # Point the Camera looks at
		          0,1,0)  # the Up-Vector
#		glutMouseFunc(mouse)
#		glutMotionFunc(motion)
#		glutSpecialFunc(keyboardListener)
		glEnable(GL_LIGHT0)
		glutDisplayFunc(self.display)
		glMatrixMode(GL_PROJECTION)
		gluPerspective(40.,1.,1.,40.)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
#		glutMainLoop()
		return

	def display(self):
		
		AT = self.AT

		for CM in self.CA:
			
			glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
#			glRotatef(rotx,1,0,0)
#			glRotatef(roty,0,1,0)
			
			for i in xrange(len(AT)):
		
				glPushMatrix()
				color = getAtomColor(AT[i])
				glTranslatef(CM[i,0],CM[i,1],CM[i,2])
				glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
				glutSolidSphere(getAtomRadius(AT[i]),20,20)
				glPopMatrix()
		
				for j in xrange(i+1):
					if self.BM[i,j] != 0:
						
						glPushMatrix()
						color = [2,2,2,2]
						dist = CM[j,:]-CM[i,:]
						norm = np.sqrt((dist).dot(dist))
						z = np.array([0,0,1],dtype=np.float)
						proj = np.cross(z,dist)
						ang = 180.0 / (np.pi) * np.arccos(np.dot(z,dist)/norm)
						quadric = gluNewQuadric()
						glTranslatef(CM[i,0],CM[i,1],CM[i,2])
						glRotatef(ang,proj[0],proj[1],proj[2])
						glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
						gluCylinder(quadric,.08,.08,norm,20,20)
						glPopMatrix()
						
#			glTranslatef(posX,posY,0)	
			glutSwapBuffers()
			glFlush()
#			time.sleep(0.1)
		
		return

def mouse(button,state,x,y):
	global beginx,beginy,rotate,translate
	if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
		print "Mouseclick: ",x,"x> ",y,"yv"
		rotate = 1
		beginx = x
		beginy = y
	if button == GLUT_LEFT_BUTTON and state == GLUT_UP:
		rotate = 0
	return


def motion(x,y):
	global rotx,roty,beginx,beginy,rotate,translate,tranx,trany
	if rotate:
		rotx += .2*(y - beginy)
		roty += .2*(x - beginx)
		beginx = x
		beginy = y
		glutPostRedisplay()
		
	return

def keyboardListener(key,foo,bar):
	
	global posX,posY
	
	if(key == GLUT_KEY_UP):
		posY = .1
		posX = 0
	elif(key == GLUT_KEY_DOWN):
		posY = -.1
		posX = 0
	elif(key == GLUT_KEY_LEFT):
		posX = -.1
		posY = 0
	elif(key == GLUT_KEY_RIGHT):
		posX = .1
		posY = 0
	else:
		if key == 27: sys.exit()
		 
	glutPostRedisplay()
	
	return

