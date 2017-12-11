/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	is_initialized = true;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	default_random_engine gen;

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(x, std_theta);


	for(int i = 0; i < num_particles; ++i)
	{
		Particle temp;
		temp.id = i;
		temp.x = dist_x(gen);
		temp.y = dist_y(gen);
		temp.theta = dist_theta(gen);
		temp.weight = 1.0f;
		particles.push_back(temp);
		weights.push_back(1.0f);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	for (int i = 0; i < num_particles; ++i)
	{
		// Prediction equations
		if (abs(yaw_rate) != 0)
		{
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }
		else
		{
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		// Add noise with zero mean and given std

  	default_random_engine gen;
		double std_x = std_pos[0];
		double std_y = std_pos[1];
		double std_theta = std_pos[2];
		normal_distribution<double> dist_x(0, std_x);
		normal_distribution<double> dist_y(0, std_y);
		normal_distribution<double> dist_theta(0, std_theta);

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	for(auto &particle : particles)
	{
		// Transform each observation from vehicle perspective to map perspective
		vector<LandmarkObs> trans_observations;
		for (auto obs : observations)
		{
			LandmarkObs temp;
			temp.x = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
			temp.y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
			trans_observations.push_back(temp);
		}

		vector<double> observation_weights;
		// Find the the nearest landmark for each observation
		for (auto t_obs : trans_observations)
		{
			double min_distance = sensor_range;
			double current_distance = 0.0f;
			double landmark_x = 0.0f;
			double landmark_y = 0.0f;

			for (auto &landmark : map_landmarks.landmark_list)
			{
				current_distance = sqrt(pow((t_obs.x - landmark.x_f), 2) + pow((t_obs.y - landmark.y_f), 2));
				if (current_distance < min_distance)
				{
					min_distance = current_distance;
					landmark_x = landmark.x_f;
					landmark_y = landmark.y_f;
				}
			}

			// calculate normalization term
			double gauss_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));

		  // calculate exponent
			double exponent = (pow((t_obs.x - landmark_x), 2)) / (2 * pow(t_obs.x, 2)) + (pow((t_obs.y - landmark_y),2)) / (2 * pow(t_obs.y,2));

		  // calculate weight using normalization terms and exponent
			observation_weights.push_back(gauss_norm * gauss_norm * exp(-exponent));
		}
		particle.weight = observation_weights[0] * observation_weights[1] * observation_weights[2];
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<double> weights;
	for(auto particle : particles)
	{
		weights.push_back(particle.weight);
	}

	// Vector for new particles
	std::vector<Particle> new_particles;

  // Use discrete distribution to return particles by weight
	default_random_engine gen;
  for (int i = 0; i < num_particles; ++i) {
    discrete_distribution<> d(weights.begin(), weights.end());
    new_particles.push_back(particles[d(gen)]);
  }

  // Replace old particles with the resampled particles
  particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
