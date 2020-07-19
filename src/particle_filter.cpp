/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Completed by Matt Weist on July 13, 2020
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
std::default_random_engine gen; // create random engine

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = 100;
  // Create normal distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  // loop through number of particles
  for (int i = 0; i < num_particles; i++) {
    // create particle object
  	Particle p;
    // initialize the states and add noise to states by sampling from normal distributions
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    // set the initial weight to 1
    p.weight = 1.0;
    // populate set of current particles
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Create normal distributions for x, y, and theta
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);
  for(int i = 0; i < num_particles; i++) {
    // calculate new position and angle
    // if yaw_rate is too low (to avoid division by zero)
    if(fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    // if yaw_rate is considerable
    else {
      particles[i].x += velocity/yaw_rate * ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta) );
      particles[i].y += velocity/yaw_rate * ( -cos(particles[i].theta + yaw_rate*delta_t) + cos(particles[i].theta) );
      particles[i].theta += yaw_rate*delta_t;
    }
    // add random noise to the calculated states
    particles[i].x += noise_x(gen);
    particles[i].y += noise_y(gen);
    particles[i].theta += noise_theta(gen); 
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {  
  // loop through observations
  for(unsigned int i = 0; i<observations.size(); i++) {
    LandmarkObs obs = observations[i];
    // create variables for minimum distance and its index
    double min_dist = 10000000000; 
    int min_id = -1;
    // loop through predictions
    for(unsigned int j = 0; j<predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      // calculate distance from measurement to landmark
      double distance = sqrt(pow(obs.x-pred.x,2) + pow(obs.y-pred.y,2));
      // replace minimum distance if current distance is the smallest encountered and save its index
      if (distance < min_dist) {
        min_dist = distance;
        min_id = pred.id;
      }
    }
    // set id of observaton to id of landmark with minimum distance
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  // loop through each particle
  for(int i = 0; i<num_particles; i++) {
    // create landmark prediction vector
    vector<LandmarkObs> lm_predict;
    // populate the landmark prediction vector
    for(unsigned int j = 0; j<map_landmarks.landmark_list.size(); j++) {
      int lm_id = map_landmarks.landmark_list[j].id_i;
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      // add landmark measurement to predictions if it is within sensor range of the particle
      if( sqrt( pow(particles[i].x-lm_x,2) + pow(particles[i].y-lm_y,2) ) <= sensor_range ) {
      	lm_predict.push_back(LandmarkObs{lm_id,lm_x,lm_y});
      }
    }
    // create observations vector for landmark observations transformed from vehicle to map frame
    vector<LandmarkObs> lm_obs;
	// loop through observations, transforming observations from car frame to map frame    
    for(unsigned int j = 0; j<observations.size(); j++) {
      double lm_obs_x = particles[i].x + cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y;
      double lm_obs_y = particles[i].y + sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y;
      lm_obs.push_back(LandmarkObs{observations[j].id, lm_obs_x, lm_obs_y});
    }
    // perform nearest neighbor function to match measurements with landmarks
    dataAssociation(lm_predict, lm_obs);
    // set weights to 1
    particles[i].weight = 1.0;
    // loop through map frame observations
    for(unsigned int j = 0; j<lm_obs.size(); j++) {
      double mu_x, mu_y, x_obs, y_obs;
      // observed position
      x_obs = lm_obs[j].x;
      y_obs = lm_obs[j].y;
      // loop through predictions
      for(unsigned int k = 0; k<lm_predict.size();k++) {
        // if the id matches up, set the landmark prediction position mu
        if(lm_predict[k].id == lm_obs[j].id) {
          // landmark position
          mu_x = lm_predict[k].x;
          mu_y = lm_predict[k].y;
        }
      }
      // standard deviations
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      // calculate observation weight using multivariate Gaussian
      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
      double exponent = ( pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) ) 
       					 + ( pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)) );
      // calculate weight for this observation
      double w_obs = gauss_norm * exp(-exponent);
      // calculate total weight for the particle
      particles[i].weight *= w_obs;
    }
  }
}

void ParticleFilter::resample() {
  // create clean vector for weights to be sampled
  vector<double> weights;
  for(int i = 0; i<num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  // max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  // create uniform distribution of indices and grab random index for resampling
  std::uniform_int_distribution<int> intdist(0, num_particles);
  int index = intdist(gen);
  // create uniform distribution of weights
  std::uniform_real_distribution<double> doubdist(0.0, max_weight);
  // intermediate variable beta
  double beta = 0.0;
  // create new vector for resampled particle set
  vector<Particle> resampledParticles;
  for(int i = 0; i<num_particles; i++) {
    beta += doubdist(gen) * 2.0;
    while(beta>weights[index]) {
      beta -= weights[index];
      index = (index+1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
