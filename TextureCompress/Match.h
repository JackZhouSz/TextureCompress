#ifndef MATCH
#define MATCH

// ===============================
// define affine deformation
// ===============================

class Match {
public:
	Match(float _m_00, float _m_01, float _m_10, float _m_11, float _m_a, float _m_b) {
		m_00 = _m_00;
		m_01 = _m_01;
		m_10 = _m_10;
		m_11 = _m_11;
		m_a  =  _m_a;
		m_b  =  _m_b;
	}
	// M(p)=(m_00,m_01  (p_x  + (m_a
	//		 m_10,m_11)	 p_y)    m_b)
	float m_00, m_01, m_10, m_11, m_a, m_b; //affine deformation parameters
	
};


#endif // !MATCH

