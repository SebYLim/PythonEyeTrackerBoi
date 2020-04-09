import turtle
import csv

coords = []
with open('SP1a_Export_001.csv') as f:
    for record in csv.DictReader(f):
        coords.append((float(record.get('X'))/3, float(record.get('Y'))/3))

turtle.goto(coords[0],coords[1])
for coord in coords:
    if(coord[0] != -1):
        turtle.goto(coord[0],coord[1])
