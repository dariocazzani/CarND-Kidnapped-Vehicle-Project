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

	num_particles = 120;
	is_initialized = true;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	default_random_engine gen;

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

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

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	default_random_engine gen;

	normal_distribution<double> dist_x(0.0f, std_x);
	normal_distribution<double> dist_y(0.0f, std_y);
	normal_distribution<double> dist_theta(0.0f, std_theta);

 	for(auto& particle: particles)
	{
		double epsilon = 0.00001f;
		if(fabs(yaw_rate) < epsilon)
		{
			particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }
		else
		{
      particle.x += velocity / yaw_rate * ( sin(particle.theta + yaw_rate*delta_t ) - sin(particle.theta) );
      particle.y += velocity / yaw_rate * ( cos(particle.theta ) - cos(particle.theta + yaw_rate*delta_t ) );
      particle.theta += yaw_rate * delta_t;
    }

		// Add noise with zero mean
 		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);
 	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for(auto& obs: observations)
	{
		double min_distance = 1000000.0f;

	  for(const auto& pred: predicted)
		{
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
	    if(min_distance > distance)
			{
				min_distance = distance;
	      obs.id = pred.id;
	    }
	  }
	}
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
  //
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	for(auto &particle : particles)
	{
	  particle.weight = 1.0;

    // Make sure that observations are withing range
    vector<LandmarkObs> predictions;
    for(const auto& landmark: map_landmarks.landmark_list)
		{
      double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
      if(distance < sensor_range)
			  predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
    }

    // Transformation step
    vector<LandmarkObs> observations_map_coor;

    for(const auto& obs: observations){
      LandmarkObs temp;
      temp.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
      temp.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
      //tmp.id = obs.id; // maybe an unnecessary step, since the each obersation will get the id from dataAssociation step.
      observations_map_coor.push_back(temp);
    }

    // data association
    dataAssociation(predictions, observations_map_coor);

    // Particles' weitghs
    for(const auto& obs_m: observations_map_coor){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);

			// calculate normalization term
			double gauss_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));

			// calculate exponent
			double exponent = (pow(obs_m.x - landmark.x_f, 2) / (2 * pow(sig_x, 2))) + (pow(obs_m.y - landmark.y_f, 2) / (2 * pow(sig_y, 2)));

      double w = exp(-exponent) * gauss_norm;
      particle.weight *=  w;
    }

    weights.push_back(particle.weight);

  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Vector for new particles
	std::vector<Particle> new_particles;

  // Use discrete distribution to return particles by weight
	default_random_engine gen;
  for(auto &particle : particles)
	{
    discrete_distribution<> d(weights.begin(), weights.end());
    new_particles.push_back(particles[d(gen)]);
  }

  // Replace old particles with the resampled particles
  particles = new_particles;

	weights.clear();

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

		return particle;
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
